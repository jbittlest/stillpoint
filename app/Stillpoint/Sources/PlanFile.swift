// .spplan v1 reader (format: engine/stillpoint/plan_io.py; same parsing as app/renderer/main.swift), including the
// engine v5 extensions: the FILL section (flags bit 0, full-frame border fill) and the MESH block (flags bit 1, mesh
// residual). Like sprender, a plan that asks for both is refused (their kernels are separate).
import Foundation

final class PlanFile: @unchecked Sendable {
    let url: URL
    let data: Data
    let headerBytes: Int
    let srcW: Int, srcH: Int, outW: Int, outH: Int
    let count: Int, nRows: Int, lensModel: UInt32, recordBytes: Int
    let lens: [Float]
    let pts: [Double]
    let frameDuration: Double
    let flags: UInt32

    /// FILL section (fill.py): per record up to K neighbouring source records + weights, gains, rotations G, and a
    /// parallax velocity grid. Parameters exactly as sprender reads them.
    struct Fill {
        let offset: Int, headerBytes: Int, k: Int, recordBytes: Int, meshW: Int, meshH: Int
        let featherMain: Float, featherNb: Float, sigma: Float, fallbackWeight: Float, fallbackBlur: Float
        let maxOffset: Int
        let parallaxTol: Float
    }
    /// MESH block (mesh.py): per record (ny x nx) float2 output-pixel offsets.
    struct Mesh {
        let nx: Int, ny: Int, offset: Int, frameBytes: Int, clampPx: Float
    }
    let fill: Fill?
    let mesh: Mesh?
    var hasFill: Bool { fill != nil }
    var hasMesh: Bool { mesh != nil }

    struct Failure: LocalizedError { let errorDescription: String? }

    init(url: URL) throws {
        self.url = url
        data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 256, String(data: data.prefix(8), encoding: .ascii) == "SPPLAN01" else {
            throw Failure(errorDescription: "Not a Stillpoint plan: \(url.lastPathComponent)")
        }
        let d = data
        func u32(_ o: Int) -> UInt32 { d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) } }
        func u64(_ o: Int) -> UInt64 { d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt64.self) } }
        func f32(_ o: Int) -> Float { d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: Float.self) } }
        func f64(_ o: Int) -> Double { d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: Double.self) } }
        guard u32(8) == 1 else { throw Failure(errorDescription: "Unsupported plan version \(u32(8))") }
        headerBytes = Int(u32(12))
        srcW = Int(u32(16)); srcH = Int(u32(20)); outW = Int(u32(24)); outH = Int(u32(28))
        count = Int(u32(32)); nRows = Int(u32(36)); lensModel = u32(40); recordBytes = Int(u32(44))
        lens = (0..<8).map { f32(48 + 4 * $0) }
        flags = u32(84)
        guard recordBytes == 24 + 36 * nRows, nRows >= 2, count > 0, data.count >= headerBytes + count * recordBytes,
              outW % 2 == 0, outH % 2 == 0 else {
            throw Failure(errorDescription: "Corrupt plan: \(url.lastPathComponent)")
        }
        let hb = headerBytes, rb = recordBytes
        pts = (0..<count).map { f64(hb + $0 * rb) }
        for i in 1..<max(count, 1) where pts[i] <= pts[i - 1] {
            throw Failure(errorDescription: "Plan PTS not increasing at record \(i)")
        }
        if count >= 2 {
            let diffs = zip(pts.dropFirst(), pts).map { $0 - $1 }.sorted()
            frameDuration = diffs[diffs.count / 2]
        } else {
            frameDuration = 1.0 / 60.0
        }
        // FILL section (flags bit 0), directly after the records
        if flags & 1 != 0 {
            let fo = hb + count * rb
            guard data.count >= fo + 64, String(data: data.subdata(in: fo..<(fo + 8)), encoding: .ascii) == "SPFILL01",
                  u32(fo + 8) == 1 else {
                throw Failure(errorDescription: "Plan announces a fill section but none was found: \(url.lastPathComponent)")
            }
            let f = Fill(offset: fo, headerBytes: Int(u32(fo + 12)), k: Int(u32(fo + 20)), recordBytes: Int(u32(fo + 24)),
                         meshW: Int(u32(fo + 28)), meshH: Int(u32(fo + 32)), featherMain: f32(fo + 36),
                         featherNb: f32(fo + 40), sigma: f32(fo + 44), fallbackWeight: f32(fo + 48),
                         fallbackBlur: f32(fo + 52), maxOffset: Int(u32(fo + 56)), parallaxTol: f32(fo + 60))
            guard Int(u32(fo + 16)) == count, f.recordBytes == 16 + 64 * f.k + 8 * f.meshW * f.meshH,
                  data.count >= fo + f.headerBytes + count * f.recordBytes else {
                throw Failure(errorDescription: "Corrupt fill section: \(url.lastPathComponent)")
            }
            fill = f
        } else {
            fill = nil
        }
        // MESH block (flags bit 1), at an explicit offset
        if flags & 2 != 0 {
            let m = Mesh(nx: Int(u32(88)), ny: Int(u32(92)), offset: Int(u64(96)), frameBytes: Int(u32(104)),
                         clampPx: f32(108))
            guard m.nx >= 2, m.ny >= 2, m.frameBytes == 8 * m.nx * m.ny, data.count >= m.offset + count * m.frameBytes else {
                throw Failure(errorDescription: "Corrupt mesh block: \(url.lastPathComponent)")
            }
            mesh = m
        } else {
            mesh = nil
        }
        if fill != nil && mesh != nil {
            throw Failure(errorDescription: "Plan has both a fill section and a mesh block, which the renderer cannot combine yet")
        }
    }

    func u32(_ o: Int) -> UInt32 { data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) } }
    func f32(_ o: Int) -> Float { data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: Float.self) } }
    func outFx(_ r: Int) -> Float { f32(headerBytes + r * recordBytes + 8) }
    func outCx(_ r: Int) -> Float { f32(headerBytes + r * recordBytes + 12) }
    func outCy(_ r: Int) -> Float { f32(headerBytes + r * recordBytes + 16) }
    /// Row matrix (row-major 3x3) `row` of record r.
    func matrix(_ r: Int, row: Int) -> [Float] {
        let b = headerBytes + r * recordBytes + 24 + 36 * row
        return (0..<9).map { f32(b + 4 * $0) }
    }

    /// Record for the source frame shown at composition time t: the last record with pts <= t (+0.1 ms rounding
    /// tolerance), provided it lies within 1.5 frames. That is exactly the frame AVFoundation displays at t.
    func record(at t: Double) -> Int? {
        var lo = 0, hi = count
        while lo < hi { let m = (lo + hi) / 2; if pts[m] <= t + 1e-4 { lo = m + 1 } else { hi = m } }
        let r = lo - 1
        guard r >= 0, t - pts[r] < 1.5 * frameDuration else { return nil }
        return r
    }

    /// Raw pointer to record r's n_rows x 3x3 float32 row-major matrices.
    func withMatrices<R>(_ r: Int, _ body: (UnsafeRawPointer, Int) -> R) -> R {
        data.withUnsafeBytes { raw in body(raw.baseAddress!.advanced(by: headerBytes + r * recordBytes + 24), 36 * nRows) }
    }

    // MARK: fill

    private func fillBase(_ r: Int) -> Int { fill!.offset + fill!.headerBytes + r * fill!.recordBytes }
    /// Number of fill sources of record r (as sprender: at most K and 4).
    func fillCount(_ r: Int) -> Int {
        guard let f = fill else { return 0 }
        return min(Int(u32(fillBase(r))), f.k, 4)
    }
    /// Fraction of record r's output the engine expects to synthesise (0 when the frame stays inside its source).
    func fillFraction(_ r: Int) -> Float { fill == nil ? 0 : f32(fillBase(r) + 4) }
    struct FillSlot { let source: Int; let weight: Float; let gain: Float; let g: [Float] }
    func fillSlot(_ r: Int, _ i: Int) -> FillSlot {
        let b = fillBase(r) + 16 + 64 * i
        return FillSlot(source: Int(Int32(bitPattern: u32(b))), weight: f32(b + 4), gain: f32(b + 8),
                        g: (0..<9).map { f32(b + 16 + 4 * $0) })
    }
    /// The parallax velocity grid of record r (mesh_h x mesh_w x 2 floats), [0, 0] when the section has none.
    func fillVelocity(_ r: Int) -> [Float] {
        guard let f = fill, f.meshW * f.meshH > 0 else { return [0, 0] }
        let b = fillBase(r) + 16 + 64 * f.k
        return (0..<(2 * f.meshW * f.meshH)).map { f32(b + 4 * $0) }
    }
    /// Plan records whose source frames record r's fill samples.
    func fillSources(_ r: Int) -> [Int] {
        (0..<fillCount(r)).map { fillSlot(r, $0).source }.filter { $0 >= 0 && $0 < count }
    }

    // MARK: mesh

    /// Record r's mesh offsets (the kernels' buffer).
    func withMesh<R>(_ r: Int, _ body: (UnsafeRawPointer, Int) -> R) -> R? {
        guard let m = mesh else { return nil }
        return data.withUnsafeBytes { raw in body(raw.baseAddress!.advanced(by: m.offset + r * m.frameBytes), m.frameBytes) }
    }
}
