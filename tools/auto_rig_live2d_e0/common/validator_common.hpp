#pragma once

#include <Model/CubismModel.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace qinglong::live2d
{
inline constexpr const char* ValidatorProtocolDigest =
    "sha256:5524ac6618c435c1e62ee9b156c1d9f1a140659d797b0446a036b0fc112c061e";
inline constexpr const char* ProbeSchemaVersion = "auto-rig-live2d-probe-v1";
inline constexpr const char* RenderSchemaVersion = "auto-rig-live2d-render-v2";

struct Options
{
    std::filesystem::path MocPath;
    std::vector<std::filesystem::path> TexturePaths;
    std::filesystem::path MotionPath;
    std::filesystem::path ExpressionPath;
    std::filesystem::path OutputPath;
    std::filesystem::path ReportPath;
    std::filesystem::path ProbeReportPath;
    unsigned int Width = 0;
    unsigned int Height = 0;
    float EvaluationTime = 0.0f;
    std::map<std::string, float> Parameters;
    std::vector<std::string> ObservedParameters;
};

struct BackendRenderResult
{
    std::vector<std::uint8_t> Rgba;
    std::map<std::string, std::string> RuntimeInfo;
};

class ValidatorBackend
{
public:
    virtual ~ValidatorBackend() = default;
    virtual const char* BackendId() const noexcept = 0;
    virtual BackendRenderResult Render(const Options& options, Csm::CubismModel* model) = 0;
};

std::vector<std::uint8_t> ReadFile(const std::filesystem::path& path);
int RunValidator(int argc, char** argv, ValidatorBackend& backend);
} // namespace qinglong::live2d
