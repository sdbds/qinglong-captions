#include <spine/Version.h>
#include <spine/spine.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spine {
SpineExtension* getDefaultExtension() {
    return new DefaultSpineExtension();
}
}  // namespace spine

namespace {

constexpr int kSampleRateHz = 60;
constexpr float kChangeTolerance = 1.0e-4F;
constexpr const char* kSchemaVersion = "auto-rig-spine-runtime-v3";
constexpr const char* kProtocolDigest =
    "sha256:fa260823ed4e0ac2028263d36132f730e2791bf0b508beae36e6e68daa950b62";

struct Options {
    std::string skeletonPath;
    std::string atlasPath;
    std::string reportPath;
};

struct Snapshot {
    std::vector<float> values;
    std::vector<float> slotAlphas;
    std::vector<float> worldVertices;
    std::vector<std::string> attachmentNames;
    bool finite = true;
};

struct AnimationEvidence {
    std::string name;
    float duration = 0.0F;
    int sampleCount = 0;
    bool finite = true;
    bool visibleChange = false;
    float maximumNumericStateDelta = 0.0F;
    float maximumWorldVertexDisplacement = 0.0F;
    float maximumWorldVertexDisplacementRatio = 0.0F;
    float maximumSlotAlphaDelta = 0.0F;
    float maximumSetupRestoreResidual = 0.0F;
};

std::string JsonEscape(const std::string& value) {
    std::ostringstream stream;
    for (const unsigned char character : value) {
        switch (character) {
            case '\"':
                stream << "\\\"";
                break;
            case '\\':
                stream << "\\\\";
                break;
            case '\b':
                stream << "\\b";
                break;
            case '\f':
                stream << "\\f";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    stream << static_cast<char>(character);
                }
        }
    }
    return stream.str();
}

Options ParseOptions(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::runtime_error("every option requires a value");
        }
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        if (key == "--skeleton") {
            options.skeletonPath = value;
        } else if (key == "--atlas") {
            options.atlasPath = value;
        } else if (key == "--report") {
            options.reportPath = value;
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    if (options.skeletonPath.empty() || options.atlasPath.empty() || options.reportPath.empty()) {
        throw std::runtime_error("--skeleton, --atlas, and --report are required");
    }
    return options;
}

class HeadlessTextureLoader final : public spine::TextureLoader {
public:
    void load(spine::AtlasPage& page, const spine::String& path) override {
        std::ifstream input(path.buffer(), std::ios::binary);
        if (!input.good()) {
            missingPaths_.emplace_back(path.buffer());
        }
        page.texture = new int(1);
    }

    void unload(void* texture) override {
        delete static_cast<int*>(texture);
    }

    const std::vector<std::string>& missingPaths() const {
        return missingPaths_;
    }

private:
    std::vector<std::string> missingPaths_;
};

void AddValue(Snapshot& snapshot, const float value) {
    snapshot.values.push_back(value);
    snapshot.finite = snapshot.finite && std::isfinite(value);
}

Snapshot CaptureSnapshot(spine::Skeleton& skeleton) {
    Snapshot snapshot;
    spine::Vector<spine::Bone*>& bones = skeleton.getBones();
    for (std::size_t index = 0; index < bones.size(); ++index) {
        spine::Bone* bone = bones[index];
        AddValue(snapshot, bone->getX());
        AddValue(snapshot, bone->getY());
        AddValue(snapshot, bone->getRotation());
        AddValue(snapshot, bone->getScaleX());
        AddValue(snapshot, bone->getScaleY());
        AddValue(snapshot, bone->getShearX());
        AddValue(snapshot, bone->getShearY());
        AddValue(snapshot, bone->getA());
        AddValue(snapshot, bone->getB());
        AddValue(snapshot, bone->getC());
        AddValue(snapshot, bone->getD());
        AddValue(snapshot, bone->getWorldX());
        AddValue(snapshot, bone->getWorldY());
    }
    spine::Vector<spine::Slot*>& slots = skeleton.getSlots();
    for (std::size_t index = 0; index < slots.size(); ++index) {
        spine::Slot* slot = slots[index];
        const spine::Color& color = slot->getColor();
        AddValue(snapshot, color.r);
        AddValue(snapshot, color.g);
        AddValue(snapshot, color.b);
        AddValue(snapshot, color.a);
        snapshot.slotAlphas.push_back(color.a);
        spine::Attachment* attachment = slot->getAttachment();
        snapshot.attachmentNames.emplace_back(
            attachment == nullptr ? "" : attachment->getName().buffer());
        if (attachment == nullptr ||
            !attachment->getRTTI().instanceOf(spine::VertexAttachment::rtti)) {
            continue;
        }
        auto* vertexAttachment = static_cast<spine::VertexAttachment*>(attachment);
        const std::size_t valueCount = vertexAttachment->getWorldVerticesLength();
        std::vector<float> worldVertices(valueCount, 0.0F);
        vertexAttachment->computeWorldVertices(*slot, worldVertices.data());
        for (const float value : worldVertices) {
            AddValue(snapshot, value);
            snapshot.worldVertices.push_back(value);
        }
    }
    return snapshot;
}

bool SnapshotChanged(const Snapshot& before, const Snapshot& after) {
    if (before.attachmentNames != after.attachmentNames ||
        before.values.size() != after.values.size()) {
        return true;
    }
    for (std::size_t index = 0; index < before.values.size(); ++index) {
        if (std::abs(before.values[index] - after.values[index]) > kChangeTolerance) {
            return true;
        }
    }
    return false;
}

float MaximumResidual(const Snapshot& expected, const Snapshot& actual) {
    if (expected.attachmentNames != actual.attachmentNames ||
        expected.values.size() != actual.values.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float maximum = 0.0F;
    for (std::size_t index = 0; index < expected.values.size(); ++index) {
        maximum = std::max(maximum, std::abs(expected.values[index] - actual.values[index]));
    }
    return maximum;
}

float MaximumScalarDelta(
    const std::vector<float>& before,
    const std::vector<float>& after) {
    if (before.size() != after.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float maximum = 0.0F;
    for (std::size_t index = 0; index < before.size(); ++index) {
        maximum = std::max(maximum, std::abs(before[index] - after[index]));
    }
    return maximum;
}

float MaximumPointDisplacement(
    const std::vector<float>& before,
    const std::vector<float>& after) {
    if (before.size() != after.size() || before.size() % 2U != 0U) {
        return std::numeric_limits<float>::infinity();
    }
    float maximum = 0.0F;
    for (std::size_t index = 0; index < before.size(); index += 2U) {
        maximum = std::max(
            maximum,
            std::hypot(before[index] - after[index], before[index + 1U] - after[index + 1U]));
    }
    return maximum;
}

float MaximumPointDisplacementRatio(
    const std::vector<float>& before,
    const std::vector<float>& after) {
    if (before.size() != after.size() || before.size() % 2U != 0U) {
        return std::numeric_limits<float>::infinity();
    }
    float maximum = 0.0F;
    float minimumX = std::numeric_limits<float>::infinity();
    float minimumY = std::numeric_limits<float>::infinity();
    float maximumX = -std::numeric_limits<float>::infinity();
    float maximumY = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < before.size(); index += 2U) {
        const float displacement = std::hypot(
            before[index] - after[index], before[index + 1U] - after[index + 1U]);
        maximum = std::max(maximum, displacement);
        if (displacement <= kChangeTolerance) {
            continue;
        }
        minimumX = std::min(minimumX, before[index]);
        minimumY = std::min(minimumY, before[index + 1U]);
        maximumX = std::max(maximumX, before[index]);
        maximumY = std::max(maximumY, before[index + 1U]);
    }
    if (maximum <= kChangeTolerance) {
        return 0.0F;
    }
    const float supportExtent = std::max(
        {maximumX - minimumX, maximumY - minimumY, 1.0F});
    return maximum / supportExtent;
}

AnimationEvidence ExerciseAnimation(spine::SkeletonData* data, spine::Animation* animation) {
    spine::Skeleton skeleton(data);
    skeleton.setToSetupPose();
    skeleton.updateWorldTransform(spine::Physics_Update);
    const Snapshot setup = CaptureSnapshot(skeleton);

    spine::AnimationStateData stateData(data);
    spine::AnimationState state(&stateData);
    state.setAnimation(0, animation, false);

    AnimationEvidence evidence;
    evidence.name = animation->getName().buffer();
    evidence.duration = animation->getDuration();
    const int frameCount = std::max(1, static_cast<int>(std::ceil(
                                           evidence.duration * kSampleRateHz)));
    evidence.sampleCount = frameCount + 1;
    evidence.finite = setup.finite;
    for (int frame = 0; frame <= frameCount; ++frame) {
        if (frame > 0) {
            const float previousTime = static_cast<float>(frame - 1) / kSampleRateHz;
            const float currentTime = std::min(
                evidence.duration, static_cast<float>(frame) / kSampleRateHz);
            state.update(std::max(0.0F, currentTime - previousTime));
        }
        state.apply(skeleton);
        skeleton.updateWorldTransform(spine::Physics_Update);
        const Snapshot sample = CaptureSnapshot(skeleton);
        evidence.finite = evidence.finite && sample.finite;
        evidence.visibleChange = evidence.visibleChange || SnapshotChanged(setup, sample);
        evidence.maximumNumericStateDelta = std::max(
            evidence.maximumNumericStateDelta,
            MaximumScalarDelta(setup.values, sample.values));
        evidence.maximumWorldVertexDisplacement = std::max(
            evidence.maximumWorldVertexDisplacement,
            MaximumPointDisplacement(setup.worldVertices, sample.worldVertices));
        evidence.maximumWorldVertexDisplacementRatio = std::max(
            evidence.maximumWorldVertexDisplacementRatio,
            MaximumPointDisplacementRatio(setup.worldVertices, sample.worldVertices));
        evidence.maximumSlotAlphaDelta = std::max(
            evidence.maximumSlotAlphaDelta,
            MaximumScalarDelta(setup.slotAlphas, sample.slotAlphas));
    }

    skeleton.setToSetupPose();
    skeleton.updateWorldTransform(spine::Physics_Update);
    const Snapshot restored = CaptureSnapshot(skeleton);
    evidence.finite = evidence.finite && restored.finite;
    evidence.maximumSetupRestoreResidual = MaximumResidual(setup, restored);
    return evidence;
}

void WriteReport(
    const Options& options,
    spine::Atlas& atlas,
    spine::SkeletonData& data,
    const std::vector<AnimationEvidence>& animations,
    const std::size_t setupAttachmentCount) {
    std::ofstream output(options.reportPath, std::ios::binary | std::ios::trunc);
    if (!output.good()) {
        throw std::runtime_error("unable to open report output");
    }
    output << std::setprecision(9);
    output << "{";
    output << "\"schema_version\":\"" << kSchemaVersion << "\",";
    output << "\"validator_protocol_digest\":\"" << kProtocolDigest << "\",";
    output << "\"runtime_version\":\"" << SPINE_VERSION_STRING << "\",";
    output << "\"skeleton_version\":\"" << JsonEscape(data.getVersion().buffer()) << "\",";
    output << "\"sample_rate_hz\":" << kSampleRateHz << ",";
    output << "\"bone_count\":" << data.getBones().size() << ",";
    output << "\"slot_count\":" << data.getSlots().size() << ",";
    output << "\"skin_count\":" << data.getSkins().size() << ",";
    output << "\"setup_attachment_count\":" << setupAttachmentCount << ",";
    output << "\"animation_count\":" << animations.size() << ",";
    output << "\"atlas_page_count\":" << atlas.getPages().size() << ",";
    output << "\"atlas_region_count\":" << atlas.getRegions().size() << ",";
    output << "\"animations\":[";
    for (std::size_t index = 0; index < animations.size(); ++index) {
        if (index > 0) {
            output << ",";
        }
        const AnimationEvidence& evidence = animations[index];
        output << "{";
        output << "\"name\":\"" << JsonEscape(evidence.name) << "\",";
        output << "\"duration\":" << evidence.duration << ",";
        output << "\"sample_count\":" << evidence.sampleCount << ",";
        output << "\"finite\":" << (evidence.finite ? "true" : "false") << ",";
        output << "\"visible_change\":"
               << (evidence.visibleChange ? "true" : "false") << ",";
        output << "\"maximum_numeric_state_delta\":"
               << evidence.maximumNumericStateDelta << ",";
        output << "\"maximum_world_vertex_displacement\":"
               << evidence.maximumWorldVertexDisplacement << ",";
        output << "\"maximum_world_vertex_displacement_ratio\":"
               << evidence.maximumWorldVertexDisplacementRatio << ",";
        output << "\"maximum_slot_alpha_delta\":"
               << evidence.maximumSlotAlphaDelta << ",";
        output << "\"maximum_setup_restore_residual\":"
               << evidence.maximumSetupRestoreResidual;
        output << "}";
    }
    output << "]}";
    output.flush();
    if (!output.good()) {
        throw std::runtime_error("unable to commit report output");
    }
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        spine::Bone::setYDown(false);

        HeadlessTextureLoader textureLoader;
        spine::Atlas atlas(options.atlasPath.c_str(), &textureLoader);
        if (!textureLoader.missingPaths().empty()) {
            throw std::runtime_error(
                "atlas texture file is missing: " + textureLoader.missingPaths().front());
        }
        if (atlas.getPages().size() == 0 || atlas.getRegions().size() == 0) {
            throw std::runtime_error("atlas has no pages or regions");
        }

        spine::SkeletonJson parser(&atlas);
        spine::SkeletonData* data = parser.readSkeletonDataFile(options.skeletonPath.c_str());
        if (data == nullptr) {
            throw std::runtime_error("skeleton parse failed: " +
                                     std::string(parser.getError().buffer()));
        }

        spine::Skeleton setupSkeleton(data);
        setupSkeleton.setToSetupPose();
        setupSkeleton.updateWorldTransform(spine::Physics_Update);
        const Snapshot setup = CaptureSnapshot(setupSkeleton);
        if (!setup.finite) {
            delete data;
            throw std::runtime_error("setup pose contains non-finite values");
        }
        std::size_t setupAttachmentCount = 0;
        spine::Vector<spine::Slot*>& setupSlots = setupSkeleton.getSlots();
        for (std::size_t index = 0; index < setupSlots.size(); ++index) {
            spine::Slot* slot = setupSlots[index];
            setupAttachmentCount += slot->getAttachment() == nullptr ? 0U : 1U;
        }

        std::vector<AnimationEvidence> evidence;
        evidence.reserve(data->getAnimations().size());
        spine::Vector<spine::Animation*>& animations = data->getAnimations();
        for (std::size_t index = 0; index < animations.size(); ++index) {
            spine::Animation* animation = animations[index];
            AnimationEvidence record = ExerciseAnimation(data, animation);
            if (!record.finite) {
                const std::string name = record.name;
                delete data;
                throw std::runtime_error("animation contains non-finite state: " + name);
            }
            if (!record.visibleChange) {
                const std::string name = record.name;
                delete data;
                throw std::runtime_error("animation has no runtime-visible state change: " + name);
            }
            if (!std::isfinite(record.maximumSetupRestoreResidual) ||
                record.maximumSetupRestoreResidual > kChangeTolerance) {
                const std::string name = record.name;
                delete data;
                throw std::runtime_error("animation does not restore setup state: " + name);
            }
            evidence.push_back(record);
        }
        if (evidence.empty()) {
            delete data;
            throw std::runtime_error("skeleton contains no animations");
        }

        WriteReport(options, atlas, *data, evidence, setupAttachmentCount);
        delete data;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
