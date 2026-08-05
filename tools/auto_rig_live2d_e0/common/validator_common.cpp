#include "validator_common.hpp"

#include <CubismFramework.hpp>
#include <ICubismAllocator.hpp>
#include <Id/CubismIdManager.hpp>
#include <Model/CubismMoc.hpp>
#include <Motion/CubismExpressionMotion.hpp>
#include <Motion/CubismExpressionMotionManager.hpp>
#include <Motion/CubismMotion.hpp>
#include <Motion/CubismMotionManager.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace fs = std::filesystem;

namespace qinglong::live2d
{
namespace
{
class Allocator final : public Csm::ICubismAllocator
{
public:
    void* Allocate(const Csm::csmSizeType size) override
    {
        return std::malloc(size);
    }

    void Deallocate(void* memory) override
    {
        std::free(memory);
    }

    void* AllocateAligned(const Csm::csmSizeType size, const Csm::csmUint32 alignment) override
    {
#ifdef _WIN32
        return _aligned_malloc(size, alignment);
#else
        void* memory = nullptr;
        return posix_memalign(&memory, alignment, size) == 0 ? memory : nullptr;
#endif
    }

    void DeallocateAligned(void* memory) override
    {
#ifdef _WIN32
        _aligned_free(memory);
#else
        std::free(memory);
#endif
    }
};

struct AlphaSummary
{
    std::uint64_t Nonzero = 0;
    unsigned int MinX = 0;
    unsigned int MinY = 0;
    unsigned int MaxX = 0;
    unsigned int MaxY = 0;
};

Allocator FrameworkAllocator;
Csm::CubismFramework::Option FrameworkOptions = {};

Csm::csmByte* LoadFrameworkFile(const std::string filePath, Csm::csmSizeInt* outSize)
{
    try
    {
        const std::vector<std::uint8_t> bytes = ReadFile(fs::path(filePath));
        Csm::csmByte* result = new Csm::csmByte[bytes.size()];
        std::copy(bytes.begin(), bytes.end(), result);
        *outSize = static_cast<Csm::csmSizeInt>(bytes.size());
        return result;
    }
    catch (...)
    {
        *outSize = 0;
        return nullptr;
    }
}

void ReleaseFrameworkFile(Csm::csmByte* bytes)
{
    delete[] bytes;
}

void CoreLog(const char* message)
{
    if (message != nullptr)
    {
        std::cerr << message << '\n';
    }
}

unsigned int ParseDimension(const std::string& value, const char* field)
{
    std::size_t consumed = 0;
    const unsigned long parsed = std::stoul(value, &consumed, 10);
    if (consumed != value.size() || parsed == 0 || parsed > 8192)
    {
        throw std::runtime_error(std::string(field) + " must be in [1, 8192]");
    }
    return static_cast<unsigned int>(parsed);
}

bool IsStableParameterId(const std::string& value)
{
    return !value.empty() && std::all_of(value.begin(), value.end(), [](const unsigned char character) {
        return std::isalnum(character) != 0 || character == '_' || character == '.' || character == '-';
    });
}

float ParseFiniteFloat(const std::string& value, const char* field)
{
    std::size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed))
    {
        throw std::runtime_error(std::string(field) + " must be finite");
    }
    return parsed;
}

Options ParseOptions(const int argc, char** argv)
{
    Options options;
    if (argc == 3 && std::string(argv[1]) == "--probe-report")
    {
        options.ProbeReportPath = fs::u8path(argv[2]);
        return options;
    }
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (index + 1 >= argc)
        {
            throw std::runtime_error("missing value for argument: " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--moc")
        {
            options.MocPath = fs::u8path(value);
        }
        else if (argument == "--texture")
        {
            options.TexturePaths.push_back(fs::u8path(value));
        }
        else if (argument == "--motion")
        {
            options.MotionPath = fs::u8path(value);
        }
        else if (argument == "--expression")
        {
            options.ExpressionPath = fs::u8path(value);
        }
        else if (argument == "--output")
        {
            options.OutputPath = fs::u8path(value);
        }
        else if (argument == "--report")
        {
            options.ReportPath = fs::u8path(value);
        }
        else if (argument == "--width")
        {
            options.Width = ParseDimension(value, "width");
        }
        else if (argument == "--height")
        {
            options.Height = ParseDimension(value, "height");
        }
        else if (argument == "--time")
        {
            options.EvaluationTime = ParseFiniteFloat(value, "time");
            if (options.EvaluationTime < 0.0f)
            {
                throw std::runtime_error("time must be non-negative");
            }
        }
        else if (argument == "--parameter")
        {
            const std::size_t separator = value.find('=');
            if (separator == std::string::npos || separator == 0 || separator + 1 >= value.size())
            {
                throw std::runtime_error("parameter must use ID=value syntax");
            }
            const std::string id = value.substr(0, separator);
            if (!IsStableParameterId(id))
            {
                throw std::runtime_error("parameter ID must be stable ASCII");
            }
            const float parameterValue = ParseFiniteFloat(value.substr(separator + 1), "parameter value");
            if (!options.Parameters.emplace(id, parameterValue).second)
            {
                throw std::runtime_error("duplicate parameter ID");
            }
        }
        else if (argument == "--observe-parameter")
        {
            if (!IsStableParameterId(value))
            {
                throw std::runtime_error("observed parameter ID must be stable ASCII");
            }
            if (std::find(options.ObservedParameters.begin(), options.ObservedParameters.end(), value) !=
                options.ObservedParameters.end())
            {
                throw std::runtime_error("duplicate observed parameter ID");
            }
            options.ObservedParameters.push_back(value);
        }
        else
        {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.MocPath.empty() || options.TexturePaths.empty() || options.OutputPath.empty() ||
        options.ReportPath.empty() || options.Width == 0 || options.Height == 0)
    {
        throw std::runtime_error("moc, texture, output, report, width, and height are required");
    }
    if (options.TexturePaths.size() > 4)
    {
        throw std::runtime_error("at most four ordered texture pages are supported");
    }
    std::sort(options.ObservedParameters.begin(), options.ObservedParameters.end());
    return options;
}

AlphaSummary SummarizeAlpha(
    const std::vector<std::uint8_t>& rgba,
    const unsigned int width,
    const unsigned int height)
{
    if (rgba.size() != static_cast<std::size_t>(width) * height * 4)
    {
        throw std::runtime_error("backend returned an invalid RGBA byte count");
    }
    AlphaSummary summary;
    summary.MinX = width;
    summary.MinY = height;
    for (unsigned int y = 0; y < height; ++y)
    {
        for (unsigned int x = 0; x < width; ++x)
        {
            if (rgba[(static_cast<std::size_t>(y) * width + x) * 4 + 3] == 0)
            {
                continue;
            }
            ++summary.Nonzero;
            summary.MinX = std::min(summary.MinX, x);
            summary.MinY = std::min(summary.MinY, y);
            summary.MaxX = std::max(summary.MaxX, x);
            summary.MaxY = std::max(summary.MaxY, y);
        }
    }
    return summary;
}

std::string JsonEscape(const std::string& value)
{
    std::ostringstream escaped;
    for (const unsigned char character : value)
    {
        switch (character)
        {
        case '"':
            escaped << "\\\"";
            break;
        case '\\':
            escaped << "\\\\";
            break;
        case '\b':
            escaped << "\\b";
            break;
        case '\f':
            escaped << "\\f";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            if (character < 0x20)
            {
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(character) << std::dec;
            }
            else
            {
                escaped << character;
            }
        }
    }
    return escaped.str();
}

void WriteBytes(const fs::path& path, const std::vector<std::uint8_t>& bytes)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream || !stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()))
    {
        throw std::runtime_error("failed to write raw RGBA evidence");
    }
}

void WriteProbeReport(const fs::path& path, const ValidatorBackend& backend)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw std::runtime_error("failed to open validator probe report");
    }
    stream << "{\"backend_id\":\"" << JsonEscape(backend.BackendId())
           << "\",\"protocol_digest\":\"" << ValidatorProtocolDigest
           << "\",\"schema_version\":\"" << ProbeSchemaVersion << "\"}";
    if (!stream)
    {
        throw std::runtime_error("failed to write validator probe report");
    }
}

void WriteRenderReport(
    const fs::path& path,
    const ValidatorBackend& backend,
    const Options& options,
    const AlphaSummary& summary,
    const std::map<std::string, float>& parameterValues,
    const std::map<std::string, std::string>& runtimeInfo)
{
    if (runtimeInfo.empty())
    {
        throw std::runtime_error("backend omitted runtime diagnostics");
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw std::runtime_error("failed to open render report");
    }
    stream << "{\"alpha_bbox\":";
    if (summary.Nonzero == 0)
    {
        stream << "null";
    }
    else
    {
        stream << '[' << summary.MinX << ',' << summary.MinY << ',' << summary.MaxX + 1 << ','
               << summary.MaxY + 1 << ']';
    }
    stream << ",\"driver_type\":\"" << JsonEscape(backend.BackendId()) << "\",\"height\":"
           << options.Height << ",\"nonzero_alpha_pixels\":" << summary.Nonzero
           << ",\"parameter_values\":{";
    bool first = true;
    stream << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto& parameter : parameterValues)
    {
        if (!first)
        {
            stream << ',';
        }
        first = false;
        stream << '\"' << JsonEscape(parameter.first) << "\":" << parameter.second;
    }
    stream << "},\"premultiplied_alpha_input\":false,\"runtime_info\":{";
    first = true;
    for (const auto& item : runtimeInfo)
    {
        if (!first)
        {
            stream << ',';
        }
        first = false;
        stream << '\"' << JsonEscape(item.first) << "\":\"" << JsonEscape(item.second) << '\"';
    }
    stream << "},\"schema_version\":\"" << RenderSchemaVersion
           << "\",\"validator_protocol_digest\":\"" << ValidatorProtocolDigest
           << "\",\"width\":" << options.Width << '}';
    if (!stream)
    {
        throw std::runtime_error("failed to write render report");
    }
}

std::map<std::string, float> ApplyModelInputs(const Options& options, Csm::CubismModel* model)
{
    for (const auto& parameter : options.Parameters)
    {
        const Csm::CubismIdHandle id = Csm::CubismFramework::GetIdManager()->GetId(parameter.first.c_str());
        model->SetParameterValue(id, parameter.second);
    }
    if (!options.MotionPath.empty())
    {
        const std::vector<std::uint8_t> motionBytes = ReadFile(options.MotionPath);
        Csm::CubismMotion* motion = Csm::CubismMotion::Create(
            reinterpret_cast<const Csm::csmByte*>(motionBytes.data()),
            static_cast<Csm::csmSizeInt>(motionBytes.size()));
        if (motion == nullptr)
        {
            throw std::runtime_error("Cubism Framework rejected the motion3 payload");
        }
        Csm::CubismMotionManager motionManager;
        motionManager.StartMotionPriority(motion, true, 1);
        motionManager.UpdateMotion(model, 0.0f);
        if (options.EvaluationTime > 0.0f)
        {
            motionManager.UpdateMotion(model, options.EvaluationTime);
        }
        motionManager.StopAllMotions();
    }
    if (!options.ExpressionPath.empty())
    {
        const std::vector<std::uint8_t> expressionBytes = ReadFile(options.ExpressionPath);
        Csm::CubismExpressionMotion* expression = Csm::CubismExpressionMotion::Create(
            reinterpret_cast<const Csm::csmByte*>(expressionBytes.data()),
            static_cast<Csm::csmSizeInt>(expressionBytes.size()));
        if (expression == nullptr)
        {
            throw std::runtime_error("Cubism Framework rejected the exp3 payload");
        }
        Csm::CubismExpressionMotionManager expressionManager;
        expressionManager.StartMotion(expression, true);
        expressionManager.UpdateMotion(model, 0.0f);
        expressionManager.StopAllMotions();
    }
    model->Update();
    std::map<std::string, float> observed;
    for (const std::string& parameterId : options.ObservedParameters)
    {
        const Csm::CubismIdHandle id = Csm::CubismFramework::GetIdManager()->GetId(parameterId.c_str());
        observed.emplace(parameterId, model->GetParameterValue(id));
    }
    return observed;
}

void ValidateTextureCount(const Options& options, const Csm::CubismModel* model)
{
    int maximumTextureIndex = -1;
    for (Csm::csmInt32 drawableIndex = 0; drawableIndex < model->GetDrawableCount(); ++drawableIndex)
    {
        const int textureIndex = model->GetDrawableTextureIndex(drawableIndex);
        if (textureIndex < 0)
        {
            throw std::runtime_error("MOC3 contains a negative drawable texture index");
        }
        maximumTextureIndex = std::max(maximumTextureIndex, textureIndex);
    }
    if (static_cast<std::size_t>(maximumTextureIndex + 1) != options.TexturePaths.size())
    {
        throw std::runtime_error("ordered texture page count does not match MOC3 texture indices");
    }
}

void RenderModel(const Options& options, ValidatorBackend& backend)
{
    FrameworkOptions.LogFunction = CoreLog;
    FrameworkOptions.LoggingLevel = Csm::CubismFramework::Option::LogLevel_Info;
    FrameworkOptions.LoadFileFunction = LoadFrameworkFile;
    FrameworkOptions.ReleaseBytesFunction = ReleaseFrameworkFile;
    if (!Csm::CubismFramework::StartUp(&FrameworkAllocator, &FrameworkOptions))
    {
        throw std::runtime_error("Cubism Framework startup failed");
    }
    Csm::CubismFramework::Initialize();

    std::vector<std::uint8_t> mocBytes = ReadFile(options.MocPath);
    Csm::CubismMoc* moc = Csm::CubismMoc::Create(
        reinterpret_cast<const Csm::csmByte*>(mocBytes.data()),
        static_cast<Csm::csmSizeInt>(mocBytes.size()),
        true);
    if (moc == nullptr)
    {
        Csm::CubismFramework::Dispose();
        throw std::runtime_error("Cubism Framework rejected the MOC3 payload");
    }
    Csm::CubismModel* model = moc->CreateModel();
    if (model == nullptr)
    {
        Csm::CubismMoc::Delete(moc);
        Csm::CubismFramework::Dispose();
        throw std::runtime_error("Cubism Framework could not create the model");
    }

    try
    {
        ValidateTextureCount(options, model);
        const std::map<std::string, float> observed = ApplyModelInputs(options, model);
        BackendRenderResult result = backend.Render(options, model);
        const AlphaSummary summary = SummarizeAlpha(result.Rgba, options.Width, options.Height);
        WriteBytes(options.OutputPath, result.Rgba);
        WriteRenderReport(options.ReportPath, backend, options, summary, observed, result.RuntimeInfo);
    }
    catch (...)
    {
        moc->DeleteModel(model);
        Csm::CubismMoc::Delete(moc);
        Csm::CubismFramework::Dispose();
        throw;
    }

    moc->DeleteModel(model);
    Csm::CubismMoc::Delete(moc);
    Csm::CubismFramework::Dispose();
}
} // namespace

std::vector<std::uint8_t> ReadFile(const fs::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
    {
        throw std::runtime_error("failed to open input file: " + path.string());
    }
    const std::streamsize size = stream.tellg();
    if (size <= 0)
    {
        throw std::runtime_error("input file is empty: " + path.string());
    }
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), size))
    {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return bytes;
}

int RunValidator(const int argc, char** argv, ValidatorBackend& backend)
{
    try
    {
        const Options options = ParseOptions(argc, argv);
        if (!options.ProbeReportPath.empty())
        {
            WriteProbeReport(options.ProbeReportPath, backend);
        }
        else
        {
            RenderModel(options, backend);
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
} // namespace qinglong::live2d
