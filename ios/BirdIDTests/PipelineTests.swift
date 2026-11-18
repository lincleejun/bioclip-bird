import XCTest
@testable import BirdID

/// The app's whole path against mobile/check.py's Python one (mobile/expected/), on the same photos.
/// Needs the converted models in mobile/build (the simulator reads the Mac's files directly).
final class PipelineTests: XCTestCase {
    func testMatchesPython() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let expDir = repo.appendingPathComponent("mobile/expected")
        let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: expDir.appendingPathComponent("expected.json"))) as! [[String: Any]]
        let models = ProcessInfo.processInfo.environment["MODELDIR"] ?? repo.appendingPathComponent("mobile/build").path   // TEST_RUNNER_MODELDIR=...
        let engine = try Engine(dir: URL(fileURLWithPath: models))
        for (i, c) in cases.enumerated() {
            let path = repo.appendingPathComponent(c["path"] as! String).standardized.path, name = URL(fileURLWithPath: path).lastPathComponent   // relative to the repo
            let photo = try XCTUnwrap(Photo(data: Data(contentsOf: URL(fileURLWithPath: path))))
            let size = c["size"] as! [Int]
            XCTAssertEqual([photo.frame.w, photo.frame.h], size, name)
            let box = try XCTUnwrap(engine.detect(photo).first, name)
            let want = c["box"] as! [Double]
            XCTAssertGreaterThan(iou(box, want), 0.95, "\(name) box \(box) vs \(want)")

            // the 224 px input, pixel by pixel
            let mine = engine.crop(photo, want).image
            let theirs = try XCTUnwrap(RGB.load(Data(contentsOf: expDir.appendingPathComponent("\(i).png"))))
            let diffs = zip(mine.px, theirs.px).map { abs(Int($0) - Int($1)) }
            let mean = Double(diffs.reduce(0, +)) / Double(diffs.count)
            print("\(name): 224 px input mean |diff| \(mean), max \(diffs.max()!), exact \(diffs.filter { $0 == 0 }.count)/\(diffs.count)")
            XCTAssertLessThan(mean, 1.0, name)

            let id = try engine.identify(photo, box: want)
            let e = try XCTUnwrap(engine.lastEmbedding), we = (c["embedding"] as! [Double]).map(Float.init)
            let cos = zip(e, we).map(*).reduce(0, +)
            let top = (c["top5"] as! [[String: Any]]).map { $0["scientific"] as! String }
            print("\(name): cosine to Python \(cos); top5 \(id.top5.map(\.scientific)) vs \(top); flying \(id.flying ?? -1) vs \(c["flying"]!)")
            XCTAssertGreaterThan(cos, 0.995, name)
            XCTAssertEqual(id.top5[0].scientific, top[0], name)
            XCTAssertGreaterThanOrEqual(Set(id.top5.map(\.scientific)).intersection(top).count, 4, name)
            if let f = id.flying { XCTAssertEqual(f > 0.5, (c["flying"] as! Double) > 0.5, name) }
        }
    }

    /// gardener's species_level thresholds: species, genus, family, unconfirmed.
    func testLevel() {
        func c(_ p: [Double]) -> [Candidate] { p.enumerated().map { Candidate(scientific: "s\($0.offset)", english: "", chinese: "", p: $0.element) } }
        let taxa = [("G1", "F1"), ("G1", "F1"), ("G2", "F1"), ("G3", "F2"), ("G4", "F3")]
        XCTAssertEqual(Engine.level(c([0.9, 0.05, 0.02, 0.02, 0.01]), taxa: taxa).0, "species")
        XCTAssertEqual(Engine.level(c([0.45, 0.3, 0.1, 0.1, 0.05]), taxa: taxa).1, "G1")
        XCTAssertEqual(Engine.level(c([0.3, 0.2, 0.2, 0.2, 0.1]), taxa: taxa).0, "family")
        XCTAssertEqual(Engine.level(c([0.25, 0.1, 0.1, 0.3, 0.25]), taxa: taxa).0, "unconfirmed")
    }
}
