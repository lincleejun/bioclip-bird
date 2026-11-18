import Foundation
import ImageIO
import Observation
import UniformTypeIdentifiers

/// Where the model files come from. Release assets are flat files, so each file inside a .mlpackage is its own
/// asset, named by its path with "/" -> "__" (mobile/README.md shows how to upload them).
enum Models {
    // ponytail: placeholder tags until the first release is cut.
    static let main = URL(string: "https://github.com/lincleejun/bioclip-bird/releases/download/mobile-v0/")!          // MIT
    static let heads = URL(string: "https://github.com/lincleejun/bioclip-bird-heads/releases/download/mobile-v0/")!   // non-commercial
    static let downloadSize = "约 0.67 GB"
    static func package(_ name: String) -> [String] {
        ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin"].map { "\(name).mlpackage/\($0)" }
    }
    static let required = package("BioCLIP") + package("Detector") + ["names.f16", "names.tsv"]
    static let optional = package("Heads")       // eye / beak / flying: shown only when present

    /// `-modelDir PATH` (a launch argument, read through UserDefaults) loads straight from a folder, e.g. mobile/build
    /// on the Mac running the simulator; otherwise Application Support/Models.
    static var dir: URL {
        if let d = UserDefaults.standard.string(forKey: "modelDir") { return URL(fileURLWithPath: d) }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Models")
    }

    static var present: Bool { required.allSatisfy { FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path) } }

    /// Wi-Fi only (PRD 4.3). The heads are optional: a failure there leaves species identification working.
    static func download(progress: @escaping (String) -> Void) async throws {
        let cfg = URLSessionConfiguration.default
        cfg.allowsCellularAccess = false
        cfg.allowsExpensiveNetworkAccess = false
        let session = URLSession(configuration: cfg)
        let jobs = required.map { ($0, main, true) } + optional.map { ($0, heads, false) }
        for (n, (path, base, needed)) in jobs.enumerated() {
            let dest = dir.appendingPathComponent(path)
            if FileManager.default.fileExists(atPath: dest.path) { continue }
            progress("下载中 \(n + 1)/\(jobs.count)：\(path)")
            do {
                let (tmp, resp) = try await session.download(from: base.appendingPathComponent(path.replacingOccurrences(of: "/", with: "__")))
                guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
                try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
                try FileManager.default.moveItem(at: tmp, to: dest)
            } catch where !needed {
                continue
            }
        }
    }
}

struct Record: Codable, Identifiable {
    var id = UUID()
    var date = Date()
    var boxes: [[Double]]            // 0..1 of the frame, largest first
    var selected: Int
    var result: Identification?      // nil: no bird found
}

/// History lives only on the phone: Documents/history/<id>.json + <id>.jpg (the photo, long edge 2048).
enum History {
    static var dir: URL {
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("history")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    static func all() -> [Record] {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "json" }
            .compactMap { try? JSONDecoder().decode(Record.self, from: Data(contentsOf: $0)) }
            .sorted { $0.date > $1.date }
    }

    static func save(_ r: Record, image: RGB) {
        try? JSONEncoder().encode(r).write(to: dir.appendingPathComponent("\(r.id).json"))
        let url = dir.appendingPathComponent("\(r.id).jpg") as CFURL
        if let d = CGImageDestinationCreateWithURL(url, UTType.jpeg.identifier as CFString, 1, nil) {
            CGImageDestinationAddImage(d, image.cgImage(), [kCGImageDestinationLossyCompressionQuality: 0.85] as CFDictionary)
            CGImageDestinationFinalize(d)
        }
    }

    static func image(_ r: Record) -> URL { dir.appendingPathComponent("\(r.id).jpg") }

    static func delete(_ r: Record) {
        for ext in ["json", "jpg"] { try? FileManager.default.removeItem(at: dir.appendingPathComponent("\(r.id).\(ext)")) }
    }
}
