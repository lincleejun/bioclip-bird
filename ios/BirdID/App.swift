import PhotosUI
import SwiftUI

@main
struct BirdIDApp: App {
    @State private var model = AppModel()
    var body: some Scene {
        WindowGroup { Home().environment(model).tint(Color(red: 0.145, green: 0.302, blue: 0.224)) }
    }
}

@MainActor @Observable
final class AppModel {
    var engine: Engine?
    var status = ""
    var busy = false
    var history = History.all()
    var current: (record: Record, photo: Photo)?
    var showing = false

    init() { Task { await load() } }

    func load() async {
        guard Models.present else { status = "需要先下载模型（\(Models.downloadSize)，仅 Wi-Fi）"; return }
        busy = true
        status = "加载模型…（第一次要编译，会慢一些）"
        do {
            engine = try await Task.detached { try Engine(dir: Models.dir) }.value
            status = engine?.heads == nil ? "眼嘴和飞行组件没装，只认鸟种" : ""
            if let p = UserDefaults.standard.string(forKey: "debugImage"), let d = FileManager.default.contents(atPath: p) {
                await identify(d)              // debug launch argument: -debugImage PATH
            }
        } catch { status = "模型加载失败：\(error.localizedDescription)" }
        busy = false
    }

    func download() async {
        busy = true
        do {
            try await Models.download { s in Task { @MainActor in self.status = s } }
            busy = false
            await load()
        } catch { status = "下载失败：\(error.localizedDescription)"; busy = false }
    }

    func identify(_ data: Data) async {
        guard let engine else { return }
        busy = true
        status = "识别中…"
        let made: (Record, Photo)? = await Task.detached {
            guard let photo = Photo(data: data) else { return nil }
            let boxes = (try? engine.detect(photo)) ?? []
            let result = boxes.first.flatMap { try? engine.identify(photo, box: $0) }
            let f = photo.frame
            let rel = boxes.map { [$0[0] / Double(f.w), $0[1] / Double(f.h), $0[2] / Double(f.w), $0[3] / Double(f.h)] }
            let r = Record(boxes: rel, selected: 0, result: result)
            History.save(r, image: f)
            return (r, photo)
        }.value
        busy = false
        guard let made else { status = "读不了这张照片"; return }
        status = ""
        current = made
        history = History.all()
        showing = true
    }

    /// Another bird in the same photo: only while the decoded photo is still in memory.
    func select(_ i: Int) async {
        guard let engine, let c = current, i != c.record.selected else { return }
        var record = c.record
        let photo = c.photo
        busy = true
        let f = photo.frame, b = record.boxes[i]
        record.result = await Task.detached { try? engine.identify(photo, box: [b[0] * Double(f.w), b[1] * Double(f.h), b[2] * Double(f.w), b[3] * Double(f.h)]) }.value
        record.selected = i
        try? JSONEncoder().encode(record).write(to: History.dir.appendingPathComponent("\(record.id).json"))
        current = (record, photo)
        history = History.all()
        busy = false
    }
}

struct Home: View {
    @Environment(AppModel.self) private var model
    @State private var pick: PhotosPickerItem?

    var body: some View {
        @Bindable var model = model
        NavigationStack {
            List {
                Section {
                    PhotosPicker(selection: $pick, matching: .images) { Label("从相册选一张", systemImage: "photo.on.rectangle") }
                        .disabled(model.engine == nil || model.busy)
                    if model.engine == nil && !Models.present && !model.busy {
                        Button("下载模型（\(Models.downloadSize)）") { Task { await model.download() } }
                    }
                    if model.busy { ProgressView(model.status) } else if !model.status.isEmpty { Text(model.status).foregroundStyle(.secondary) }
                }
                Section("历史") {
                    ForEach(model.history) { r in
                        NavigationLink { ResultView(record: r, image: UIImage(contentsOfFile: History.image(r).path)) } label: { HistoryRow(record: r) }
                    }
                    .onDelete { idx in idx.forEach { History.delete(model.history[$0]) }; model.history = History.all() }
                    if !model.history.isEmpty {
                        Button("清空历史", role: .destructive) { model.history.forEach(History.delete); model.history = History.all() }
                    }
                }
            }
            .navigationTitle("观羽")
            .navigationDestination(isPresented: $model.showing) {
                if let c = model.current {
                    ResultView(record: c.record, image: UIImage(cgImage: c.photo.frame.cgImage())) { i in Task { await model.select(i) } }
                }
            }
            .onChange(of: pick) {
                guard let pick else { return }
                Task {
                    if let d = try? await pick.loadTransferable(type: Data.self) { await model.identify(d) }
                    self.pick = nil
                }
            }
        }
    }
}

struct HistoryRow: View {
    let record: Record
    var body: some View {
        HStack {
            if let img = UIImage(contentsOfFile: History.image(record).path) {
                Image(uiImage: img).resizable().scaledToFill().frame(width: 48, height: 48).clipShape(RoundedRectangle(cornerRadius: 6))
            }
            VStack(alignment: .leading) {
                Text(record.result.map(title) ?? "没找到鸟")
                Text(record.date.formatted(date: .abbreviated, time: .shortened)).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

func title(_ r: Identification) -> String {
    switch r.level {
    case "species": let c = r.top5[0]; return c.chinese.isEmpty ? c.english : c.chinese
    case "genus": return "\(r.name ?? "")属（种不确定）"
    case "family": return "\(r.name ?? "")科（种不确定）"
    default: return "不确定"
    }
}

struct ResultView: View {
    let record: Record
    let image: UIImage?
    var onSelect: ((Int) -> Void)?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let image { photo(image) }
                if let r = record.result { info(r) } else { Text("这张照片里没找到鸟。").font(.title3) }
            }
            .padding()
        }
        .navigationTitle(record.result.map(title) ?? "结果")
        .navigationBarTitleDisplayMode(.inline)
    }

    func photo(_ image: UIImage) -> some View {
        Image(uiImage: image).resizable().scaledToFit()
            .overlay {
                GeometryReader { g in
                    let s = g.size
                    ForEach(record.boxes.indices, id: \.self) { i in
                        let b = record.boxes[i]
                        Rectangle().stroke(i == record.selected ? Color.yellow : Color.white.opacity(0.7), lineWidth: i == record.selected ? 3 : 1.5)
                            .contentShape(Rectangle())
                            .frame(width: (b[2] - b[0]) * s.width, height: (b[3] - b[1]) * s.height)
                            .position(x: (b[0] + b[2]) / 2 * s.width, y: (b[1] + b[3]) / 2 * s.height)
                            .onTapGesture { onSelect?(i) }
                    }
                    if let e = record.result?.eye { dot(e, s, .red) }
                    if let k = record.result?.beak { dot(k, s, .cyan) }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    func dot(_ p: [Double], _ s: CGSize, _ c: Color) -> some View {
        Circle().stroke(c, lineWidth: 2.5).frame(width: 14, height: 14).position(x: p[0] * s.width, y: p[1] * s.height)
    }

    @ViewBuilder func info(_ r: Identification) -> some View {
        let c = r.top5[0]
        VStack(alignment: .leading, spacing: 4) {
            if r.level == "species" {
                if !c.chinese.isEmpty { Text(c.chinese).font(.largeTitle.bold()) }
                Text(c.english).font(c.chinese.isEmpty ? .largeTitle.bold() : .title3)
                Text(c.scientific).italic().foregroundStyle(.secondary)
                Text("把握 \(Int((c.p * 100).rounded()))%").font(.callout)
            } else {
                Text(title(r)).font(.largeTitle.bold())
                Text("把握不够，不给具体鸟种。最像的是 \(c.english)（\(Int((c.p * 100).rounded()))%）").font(.callout).foregroundStyle(.secondary)
            }
        }
        if record.boxes.count > 1 && onSelect != nil {
            Text("照片里有 \(record.boxes.count) 只鸟，点框换一只。").font(.footnote).foregroundStyle(.secondary)
        }
        if let f = r.flying {
            LabeledContent("在飞吗", value: f > 0.5 ? "在飞" : "没在飞")
            LabeledContent("眼 / 嘴", value: "\(r.eye == nil ? "看不到眼" : "红圈是眼") · \(r.beak == nil ? "看不到嘴" : "青圈是嘴")")
        } else {
            Text("眼嘴和飞行组件没装。").font(.footnote).foregroundStyle(.secondary)
        }
        DisclosureGroup("其他可能") {
            ForEach(Array((r.level == "species" ? r.top5.dropFirst() : r.top5[...]).enumerated()), id: \.offset) { _, c in
                HStack {
                    VStack(alignment: .leading) {
                        Text(c.chinese.isEmpty ? c.english : "\(c.chinese) · \(c.english)")
                        Text(c.scientific).italic().font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("\(Int((c.p * 100).rounded()))%").monospacedDigit()
                }
            }
        }
    }
}
