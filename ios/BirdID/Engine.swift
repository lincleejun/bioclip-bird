import Accelerate
import CoreML
import Foundation

/// One photo, decoded the way gardener's decode does: `frame` has a long edge <= 2048 (boxes live in its pixels),
/// `detail` a long edge <= 3072 when the photo is larger than the frame (species crops are cut from it).
struct Photo {
    let frame: RGB
    let detail: RGB?

    init?(data: Data) {
        guard let full = RGB.load(data) else { return nil }
        frame = full.fit(2048)
        detail = max(full.w, full.h) > max(frame.w, frame.h) ? full.fit(3072) : nil
    }
}

struct Candidate: Codable, Hashable {
    var scientific: String, english: String, chinese: String, p: Double
}

struct Identification: Codable {
    var top5: [Candidate]
    var level: String            // species / genus / family / unconfirmed (gardener rules.species_level)
    var name: String?            // the species, genus or family named at that level
    var flying: Double?          // nil without the heads file
    var beak: [Double]?          // points in the frame, 0..1 of its width / height; nil when not visible or no heads
    var eye: [Double]?
}

final class Engine: @unchecked Sendable {
    static let threshold: Float = 0.25           // D-FINE bird score; T/2 when nothing reaches it (distill/detectors.py)
    static let logitScale: Float = 98.8644790649414
    let detector: MLModel, bioclip: MLModel, heads: MLModel?
    let names: [Float]                           // 11,172 x 1024, row-major
    let rows: [[String]]                         // scientific, English, Chinese, genus, family
    private(set) var lastEmbedding: [Float]?     // for BirdIDTests

    init(dir: URL) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        #if targetEnvironment(simulator)
        cfg.computeUnits = .cpuOnly
        #endif
        detector = try MLModel(contentsOf: try Self.compiled(dir, "Detector"), configuration: cfg)
        bioclip = try MLModel(contentsOf: try Self.compiled(dir, "BioCLIP"), configuration: cfg)
        heads = FileManager.default.fileExists(atPath: dir.appendingPathComponent("Heads.mlpackage").path)
            ? try MLModel(contentsOf: try Self.compiled(dir, "Heads"), configuration: cfg) : nil
        let raw = try Data(contentsOf: dir.appendingPathComponent("names.f16"))
        names = raw.withUnsafeBytes { $0.bindMemory(to: Float16.self).map(Float.init) }
        rows = try String(contentsOf: dir.appendingPathComponent("names.tsv"), encoding: .utf8)
            .split(separator: "\n").map { $0.split(separator: "\t", omittingEmptySubsequences: false).map(String.init) }
        precondition(names.count == rows.count * 1024, "names.f16 and names.tsv disagree")
    }

    /// A .mlpackage is compiled on the device once; the compiled copy is kept in Caches.
    static func compiled(_ dir: URL, _ name: String) throws -> URL {
        let fm = FileManager.default
        let cache = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("\(name).mlmodelc")
        let pkg = dir.appendingPathComponent("\(name).mlpackage")
        let stamp = cache.appendingPathComponent("source-date")
        let date = String(describing: (try? fm.attributesOfItem(atPath: pkg.appendingPathComponent("Manifest.json").path)[.modificationDate]) ?? "")
        if (try? String(contentsOf: stamp, encoding: .utf8)) == date { return cache }
        let tmp = try MLModel.compileModel(at: pkg)
        try? fm.removeItem(at: cache)
        try fm.moveItem(at: tmp, to: cache)
        try date.write(to: stamp, atomically: true, encoding: .utf8)
        return cache
    }

    /// Bird boxes in frame pixels (x0 y0 x1 y1), largest first.
    func detect(_ photo: Photo) throws -> [[Double]] {
        let f = photo.frame
        let out = try detector.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: f.resized(640, 640, .bilinear).pixelBuffer())]))
        let scores = floats(out.featureValue(for: "scores")!.multiArrayValue!)
        let boxes = floats(out.featureValue(for: "boxes")!.multiArrayValue!)
        for t in [Self.threshold, Self.threshold / 2] {
            var keep: [[Double]] = []                // best score first; a box overlapping a kept one by IoU > 0.5 is the same bird
            for i in scores.indices.filter({ scores[$0] >= t }).sorted(by: { scores[$0] > scores[$1] }) {
                let b = (0..<4).map { Double(boxes[i * 4 + $0]) * Double($0 % 2 == 0 ? f.w : f.h) }
                if keep.allSatisfy({ iou($0, b) <= 0.5 }) { keep.append(b) }
            }
            if !keep.isEmpty { return keep.sorted { area($0) > area($1) } }
        }
        return []
    }

    /// What BioCLIP sees for one box (gardener species_crops + open_clip's framing), and where that crop sits.
    func crop(_ photo: Photo, _ box: [Double]) -> (image: RGB, origin: (Double, Double), size: (Double, Double), source: RGB) {
        var src = photo.frame, b = box, minSide = 320.0
        if let d = photo.detail, d.w != src.w || d.h != src.h {
            let sx = Double(d.w) / Double(src.w), sy = Double(d.h) / Double(src.h)
            b = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
            minSide = (320 * sx).rounded(.toNearestOrEven)
            src = d
        }
        let (x0, y0, x1, y1) = cropWithContext(src.w, src.h, b, minSide: minSide)
        return (src.cropped(x0, y0, x1, y1).frame224(), (Double(x0), Double(y0)), (Double(x1 - x0), Double(y1 - y0)), src)
    }

    func identify(_ photo: Photo, box: [Double]) throws -> Identification {
        let c = crop(photo, box)
        let out = try bioclip.prediction(from: try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: c.image.pixelBuffer())]))
        let e = floats(out.featureValue(for: "embedding")!.multiArrayValue!)
        lastEmbedding = e
        var logits = [Float](repeating: 0, count: rows.count)
        vDSP_mmul(names, 1, e, 1, &logits, 1, vDSP_Length(rows.count), 1, 1024)
        let m = logits.max()!
        var p = logits.map { exp(Self.logitScale * ($0 - m)) }
        let sum = p.reduce(0, +)
        p = p.map { $0 / sum }
        let top = p.indices.sorted { p[$0] > p[$1] }.prefix(5)
        let cands = top.map { Candidate(scientific: rows[$0][0], english: rows[$0][1], chinese: rows[$0][2], p: Double(p[$0])) }
        let (level, name) = Self.level(cands, taxa: top.map { (rows[$0][3], rows[$0][4]) })
        var id = Identification(top5: cands, level: level, name: name)
        if let heads {
            let h = try heads.prediction(from: try MLDictionaryFeatureProvider(dictionary: ["tokens15": out.featureValue(for: "tokens15")!]))
            let pts = Self.points(h)
            id.flying = Double(floats(h.featureValue(for: "flying")!.multiArrayValue!)[0])
            // 224 frame -> crop -> source image -> 0..1 (the same for frame and detail: one is a scaled copy of the other)
            let rw = c.size.0 <= c.size.1 ? 224 : Double(Int(224 * c.size.0 / c.size.1)), rh = c.size.0 <= c.size.1 ? Double(Int(224 * c.size.1 / c.size.0)) : 224
            let left = ((rw - 224) / 2).rounded(.toNearestOrEven), top = ((rh - 224) / 2).rounded(.toNearestOrEven)
            let place = { (q: [Double]) -> [Double] in
                [((q[0] + left) * c.size.0 / rw + c.origin.0) / Double(c.source.w), ((q[1] + top) * c.size.1 / rh + c.origin.1) / Double(c.source.h)]
            }
            if pts.visible[0] > 0.5 { id.beak = place(pts.xy[0]) }
            let eye = pts.visible[1] >= pts.visible[2] ? 1 : 2      // the more confident eye (parts.py score())
            if pts.visible[eye] > 0.5 { id.eye = place(pts.xy[eye]) }
        }
        return id
    }

    /// Beak / left eye / right eye in the 224 px frame and their visible probabilities.
    static func points(_ h: MLFeatureProvider) -> (xy: [[Double]], visible: [Double]) {
        let heat = floats(h.featureValue(for: "heat")!.multiArrayValue!)          // 256 x 3
        let off = floats(h.featureValue(for: "offset")!.multiArrayValue!)         // 256 x 3 x 2
        let vis = floats(h.featureValue(for: "visible")!.multiArrayValue!)        // 3
        let xy = (0..<3).map { k -> [Double] in
            let cell = (0..<256).max { heat[$0 * 3 + k] < heat[$1 * 3 + k] }!
            return [(Double(cell % 16) + Double(off[(cell * 3 + k) * 2])) * 14, (Double(cell / 16) + Double(off[(cell * 3 + k) * 2 + 1])) * 14]
        }
        return (xy, vis.map { 1 / (1 + exp(-Double($0))) })
    }

    /// gardener rules.species_level without the location prior: species when clear, else the genus or family
    /// the top five agree on, else unconfirmed.
    static func level(_ c: [Candidate], taxa: [(String, String)]) -> (String, String?) {
        let p1 = c.first?.p ?? 0, p2 = c.count > 1 ? c[1].p : 0
        if p1 >= 0.5 && p1 - p2 >= 0.3 { return ("species", c[0].scientific) }
        for (level, k) in [("genus", 0), ("family", 1)] {
            var mass: [String: Double] = [:]
            for (cand, t) in zip(c, taxa) where !(k == 0 ? t.0 : t.1).isEmpty { mass[k == 0 ? t.0 : t.1, default: 0] += cand.p }
            if let best = mass.max(by: { $0.value < $1.value }), best.value >= 0.6 { return (level, best.key) }
        }
        return ("unconfirmed", nil)
    }
}

/// Float32 or float16 outputs (which one a model gives depends on its precision) as [Float].
func floats(_ a: MLMultiArray) -> [Float] {
    a.dataType == .float16 ? MLShapedArray<Float16>(a).scalars.map(Float.init) : MLShapedArray<Float>(a).scalars
}

func area(_ b: [Double]) -> Double { (b[2] - b[0]) * (b[3] - b[1]) }

func iou(_ a: [Double], _ b: [Double]) -> Double {
    let w = max(0, min(a[2], b[2]) - max(a[0], b[0])), h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    let u = area(a) + area(b) - w * h
    return u > 0 ? w * h / u : 0
}
