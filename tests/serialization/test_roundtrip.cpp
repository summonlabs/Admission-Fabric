#include "test_util.hpp"
#include "admission_fabric/codec.hpp"
#include "admission_fabric/decision.hpp"
#include "admission_fabric/descriptor.hpp"
#include "admission_fabric/serialize.hpp"
#include "admission_fabric/policy.hpp"

using namespace admission_fabric;

// Deterministic xorshift64 RNG (prints seed).
static std::uint64_t g_seed = 0xC0FFEEu;
static void set_seed(std::uint64_t s) { g_seed = s; }
static std::uint64_t rnd64() {
    std::uint64_t x = g_seed;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    g_seed = x;
    return x;
}
static std::uint64_t rnd_bounded(std::uint64_t lo, std::uint64_t hi) { return lo + (rnd64() % (hi - lo + 1)); }

AF_TEST(serialize_primitive_roundtrip) {
    set_seed(0xDEADBEEF);
    const std::size_t kIter = 2000;
    ByteWriter w;
    for (std::size_t i = 0; i < kIter; ++i) {
        w.u64(rnd64());
        w.u32(static_cast<std::uint32_t>(rnd64()));
        w.i32(static_cast<std::int32_t>(rnd64()));
        w.f64(static_cast<double>(rnd64()) / 1e9);
        w.flag((rnd64() & 1) != 0);
        w.string("str-" + std::to_string(rnd_bounded(0, 1000)));
    }
    ByteReader r(w.data());
    set_seed(0xDEADBEEF);
    for (std::size_t i = 0; i < kIter; ++i) {
        AF_CHECK(r.u64() == rnd64());
        AF_CHECK(r.u32() == static_cast<std::uint32_t>(rnd64()));
        AF_CHECK(r.i32() == static_cast<std::int32_t>(rnd64()));
        // f64 is lossless bitwise here.
        std::uint64_t b = rnd64(); double expected = static_cast<double>(b) / 1e9;
        double got = r.f64();
        AF_CHECK(std::memcmp(&expected, &got, sizeof(double)) == 0);
        AF_CHECK(r.flag() == ((rnd64() & 1) != 0));
        std::string s; AF_CHECK(r.string(s));
        AF_CHECK(s == "str-" + std::to_string(rnd_bounded(0, 1000)));
    }
    AF_CHECK(r.remaining() == 0);
}

AF_TEST(serialize_frame_roundtrip_and_corruption) {
    ByteWriter w;
    for (int i = 0; i < 100; ++i) w.u64(static_cast<std::uint64_t>(i * 7919));
    auto framed = ByteWriter::frame(0x1234u, 1u, w.data());
    std::vector<std::uint8_t> body;
    std::string err;
    AF_CHECK(ByteReader::parse_frame(framed, 0x1234u, 1u, body, err));
    AF_CHECK(body.size() == w.data().size());
    // Corrupt a byte -> reject.
    framed[framed.size() / 2] ^= 0xFF;
    AF_CHECK(!ByteReader::parse_frame(framed, 0x1234u, 1u, body, err));
    // Wrong version -> reject.
    auto f2 = ByteWriter::frame(0x1234u, 7u, w.data());
    AF_CHECK(!ByteReader::parse_frame(f2, 0x1234u, 1u, body, err));
}

AF_TEST(codec_descriptor_roundtrip) {
    set_seed(0x1234);
    for (int k = 0; k < 500; ++k) {
        AdmissionDescriptor d;
        d.admission_request_id = AdmissionRequestId(rnd_bounded(1, 1ull << 40));
        d.request_id = RequestId(rnd_bounded(1, 1ull << 40));
        d.attempt_id = AttemptId(rnd_bounded(1, 1000));
        d.tenant_id = TenantId(rnd_bounded(1, 1u << 20));
        d.model_id = ModelId(rnd_bounded(1, 1u << 20));
        d.model_revision = ModelRevision(rnd_bounded(1, 1u << 16));
        if ((rnd64() & 1)) d.adapter_id = AdapterId(rnd_bounded(1, 1u << 10));
        d.policy_generation = PolicyGeneration(rnd_bounded(1, 1u << 16));
        d.workload_type = static_cast<WorkloadType>(rnd_bounded(0, 6));
        d.input_tokens = rnd_bounded(1, 1u << 16);
        d.expected_output_tokens = rnd_bounded(1, 1u << 16);
        d.max_output_tokens = d.expected_output_tokens + rnd_bounded(0, 100);
        d.batchable = (rnd64() & 1) != 0;
        d.priority = static_cast<Priority>(rnd_bounded(0, 63));
        d.latency_class = static_cast<LatencyClass>(rnd_bounded(0, 4));
        if ((rnd64() & 1)) d.deadline = Milliseconds(rnd_bounded(1, 10000));
        if ((rnd64() & 1)) d.requested_slo = static_cast<double>(rnd_bounded(1, 100)) / 10.0;
        d.backend = "cuda";
        d.dtype = "fp16";
        d.workload_class = "chat";
        d.policy_tags = {"a", "b", "c"};

        ByteWriter w;
        Status s = encode_descriptor(w, d);
        AF_CHECK(s.ok());
        AdmissionDescriptor d2;
        ByteReader rd(w.data());
        Status s2 = decode_descriptor(rd, d2);
        AF_CHECK(s2.ok());
        AF_CHECK(d2.admission_request_id == d.admission_request_id);
        AF_CHECK(d2.request_id == d.request_id);
        AF_CHECK(d2.attempt_id == d.attempt_id);
        AF_CHECK(d2.tenant_id == d.tenant_id);
        AF_CHECK(d2.model_id == d.model_id);
        AF_CHECK(d2.model_revision == d.model_revision);
        AF_CHECK(d2.input_tokens == d.input_tokens);
        AF_CHECK(d2.expected_output_tokens == d.expected_output_tokens);
        AF_CHECK(d2.max_output_tokens == d.max_output_tokens);
        AF_CHECK(d2.backend == d.backend);
        AF_CHECK(d2.policy_tags == d.policy_tags);
        AF_CHECK(d2.deadline == d.deadline);
    }
}

AF_TEST(codec_policy_roundtrip) {
    AdmissionPolicy p;
    p.generation = PolicyGeneration(5);
    p.name = "prod";
    p.safety_headroom_ratio = 0.15;
    p.absolute_safety_margin.set(ResourceId::AcceleratorMemory, 1u << 20);
    p.overcommit = OvercommitMode::Bounded;
    p.overcommit_allowance.set(ResourceId::AcceleratorMemory, 1u << 30);
    p.overcommit_confidence_min = 0.9;
    p.min_predictor_confidence = 0.3;
    p.max_snapshot_age = Seconds(10);
    p.require_reservation = true;
    PolicyOverride o; o.key = "tenant=3"; o.safety_headroom_ratio = 0.25; o.require_reservation = true;
    p.overrides.push_back(o);

    ByteWriter w;
    AF_CHECK(encode_policy(w, p).ok());
    AdmissionPolicy p2;
    ByteReader rd(w.data());
    AF_CHECK(decode_policy(rd, p2).ok());
    AF_CHECK(p2.generation == p.generation);
    AF_CHECK(p2.name == p.name);
    AF_CHECK(p2.safety_headroom_ratio == p.safety_headroom_ratio);
    AF_CHECK(p2.overcommit == p.overcommit);
    AF_CHECK(p2.absolute_safety_margin.get(ResourceId::AcceleratorMemory) == (1u << 20));
    AF_CHECK(p2.overrides.size() == 1);
    AF_CHECK(p2.overrides[0].key == "tenant=3");
    AF_CHECK(*p2.overrides[0].safety_headroom_ratio == 0.25);
}

AF_TEST(codec_rejects_truncation) {
    ByteWriter w;
    AdmissionDescriptor d;
    d.admission_request_id = AdmissionRequestId(1);
    encode_descriptor(w, d);
    auto bytes = w.data();
    if (bytes.size() > 4) {
        std::vector<std::uint8_t> truncated(bytes.begin(), bytes.end() - 4);
        AdmissionDescriptor d2;
        ByteReader rt(truncated);
        Status s = decode_descriptor(rt, d2);
        AF_CHECK(!s.ok());
    }
}

// Regression: decoders reject enum values beyond the last enumerator, for every
// decoded enum, and still accept the last valid one.
AF_TEST(codec_rejects_out_of_range_enums) {
    auto descriptor_ok = [](const AdmissionDescriptor& d) {
        ByteWriter w;
        Status e = encode_descriptor(w, d);
        AdmissionDescriptor out;
        ByteReader r(w.data());
        return e.ok() && decode_descriptor(r, out).ok();
    };
    AdmissionDescriptor d;
    d.admission_request_id = AdmissionRequestId(1);
    d.workload_type = WorkloadType::Other;
    d.latency_class = LatencyClass::SoftRealtime;
    d.preemption = PreemptionPermission::Deferrable;
    AF_CHECK(descriptor_ok(d));
    { AdmissionDescriptor x = d; x.workload_type = static_cast<WorkloadType>(200); AF_CHECK(!descriptor_ok(x)); }
    { AdmissionDescriptor x = d; x.latency_class = static_cast<LatencyClass>(200); AF_CHECK(!descriptor_ok(x)); }
    { AdmissionDescriptor x = d; x.preemption = static_cast<PreemptionPermission>(200); AF_CHECK(!descriptor_ok(x)); }

    auto snapshot_ok = [](const ResourceSnapshot& s) {
        ByteWriter w;
        Status e = encode_snapshot(w, s);
        ResourceSnapshot out;
        ByteReader r(w.data());
        return e.ok() && decode_snapshot(r, out).ok();
    };
    ResourceSnapshot s;
    for (auto& p : s.pressure) p = PressureLevel::Critical;
    for (auto& p : s.provenance) p = Provenance::Unavailable;
    AF_CHECK(snapshot_ok(s));
    { ResourceSnapshot x = s; x.pressure[3] = static_cast<PressureLevel>(200); AF_CHECK(!snapshot_ok(x)); }
    { ResourceSnapshot x = s; x.provenance[3] = static_cast<Provenance>(200); AF_CHECK(!snapshot_ok(x)); }

    auto policy_ok = [](const AdmissionPolicy& p) {
        ByteWriter w;
        Status e = encode_policy(w, p);
        AdmissionPolicy out;
        ByteReader r(w.data());
        return e.ok() && decode_policy(r, out).ok();
    };
    AdmissionPolicy pol;
    pol.overcommit = OvercommitMode::PredictionAware;
    PolicyOverride ov;
    ov.key = "tenant=3";
    ov.overcommit_mode = OvercommitMode::Bounded;
    pol.overrides.push_back(ov);
    AF_CHECK(policy_ok(pol));
    { AdmissionPolicy x = pol; x.overcommit = static_cast<OvercommitMode>(200); AF_CHECK(!policy_ok(x)); }
    { AdmissionPolicy x = pol; x.overrides[0].overcommit_mode = static_cast<OvercommitMode>(200); AF_CHECK(!policy_ok(x)); }

    auto decision_ok = [](const AdmissionDecision& dec) {
        ByteWriter w;
        Status e = encode_decision(w, dec);
        AdmissionDecision out;
        ByteReader r(w.data());
        return e.ok() && decode_decision(r, out).ok();
    };
    AdmissionDecision dec;
    dec.outcome = AdmissionOutcome::Reject;
    dec.reason = ReasonCode::RejectStale;
    AF_CHECK(decision_ok(dec));
    { AdmissionDecision x = dec; x.outcome = static_cast<AdmissionOutcome>(200); AF_CHECK(!decision_ok(x)); }
    { AdmissionDecision x = dec; x.reason = static_cast<ReasonCode>(999); AF_CHECK(!decision_ok(x)); }
    { AdmissionDecision x = dec; x.reason = static_cast<ReasonCode>(-1); AF_CHECK(!decision_ok(x)); }

    auto observation_ok = [](const Observation& o) {
        ByteWriter w;
        Status e = encode_observation(w, o);
        Observation out;
        ByteReader r(w.data());
        return e.ok() && decode_observation(r, out).ok();
    };
    Observation o;
    o.workload_type = WorkloadType::Other;
    o.kind = ObservationKind::Aborted;
    AF_CHECK(observation_ok(o));
    { Observation x = o; x.kind = static_cast<ObservationKind>(200); AF_CHECK(!observation_ok(x)); }
    { Observation x = o; x.workload_type = static_cast<WorkloadType>(200); AF_CHECK(!observation_ok(x)); }
}

AF_TEST_MAIN("admission-fabric-roundtrip");
