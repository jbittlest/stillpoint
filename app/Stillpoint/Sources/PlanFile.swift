// .spplan v1 reader (format: engine/stillpoint/plan_io.py; same parsing as app/renderer/main.swift).
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

    struct Failure: LocalizedError { let errorDescription: String? }

    init(url: URL) throws {
        self.url = url
        data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 256, String(data: data.prefix(8), encoding: .ascii) == "SPPLAN01" else {
            throw Failure(errorDescription: "Not a Stillpoint plan: \(url.lastPathComponent)")
        }
        let d = data
        func u32(_ o: Int) -> UInt32 { d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) } }
        func f32(_ o: Int) -> Float { d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: Float.self) } }
        func f64(_ o: Int) -> Double { d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: Double.self) } }
        guard u32(8) == 1 else { throw Failure(errorDescription: "Unsupported plan version \(u32(8))") }
        headerBytes = Int(u32(12))
        srcW = Int(u32(16)); srcH = Int(u32(20)); outW = Int(u32(24)); outH = Int(u32(28))
        count = Int(u32(32)); nRows = Int(u32(36)); lensModel = u32(40); recordBytes = Int(u32(44))
        lens = (0..<8).map { f32(48 + 4 * $0) }
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
    }

    func f32(_ o: Int) -> Float { data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: Float.self) } }
    func outFx(_ r: Int) -> Float { f32(headerBytes + r * recordBytes + 8) }
    func outCx(_ r: Int) -> Float { f32(headerBytes + r * recordBytes + 12) }
    func outCy(_ r: Int) -> Float { f32(headerBytes + r * recordBytes + 16) }

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
}
