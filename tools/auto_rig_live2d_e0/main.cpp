#include <Windows.h>
#include <d3d11.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <CubismFramework.hpp>
#include <ICubismAllocator.hpp>
#include <Id/CubismIdManager.hpp>
#include <Math/CubismMatrix44.hpp>
#include <Model/CubismMoc.hpp>
#include <Model/CubismModel.hpp>
#include <Motion/CubismExpressionMotion.hpp>
#include <Motion/CubismExpressionMotionManager.hpp>
#include <Motion/CubismMotion.hpp>
#include <Motion/CubismMotionManager.hpp>
#include <Rendering/CubismRenderer.hpp>
#include <Rendering/D3D11/CubismDeviceInfo_D3D11.hpp>
#include <Rendering/D3D11/CubismRenderer_D3D11.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

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
        return _aligned_malloc(size, alignment);
    }

    void DeallocateAligned(void* memory) override
    {
        _aligned_free(memory);
    }
};

Allocator FrameworkAllocator;
Csm::CubismFramework::Option FrameworkOptions = {};

struct Options
{
    fs::path MocPath;
    fs::path TexturePath;
    fs::path MotionPath;
    fs::path ExpressionPath;
    fs::path OutputPath;
    fs::path ReportPath;
    unsigned int Width = 0;
    unsigned int Height = 0;
    float EvaluationTime = 0.0f;
    std::map<std::string, float> Parameters;
    std::vector<std::string> ObservedParameters;
};

struct AlphaSummary
{
    std::uint64_t Nonzero = 0;
    unsigned int MinX = 0;
    unsigned int MinY = 0;
    unsigned int MaxX = 0;
    unsigned int MaxY = 0;
};

struct RenderResult
{
    std::vector<std::uint8_t> Rgba;
    std::map<std::string, float> ParameterValues;
};

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
            options.TexturePath = fs::u8path(value);
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
    if (options.MocPath.empty() || options.TexturePath.empty() || options.OutputPath.empty() ||
        options.ReportPath.empty() || options.Width == 0 || options.Height == 0)
    {
        throw std::runtime_error("moc, texture, output, report, width, and height are required");
    }
    std::sort(options.ObservedParameters.begin(), options.ObservedParameters.end());
    return options;
}

ComPtr<ID3D11ShaderResourceView> LoadTexture(
    ID3D11Device* device,
    IWICImagingFactory* factory,
    const fs::path& path)
{
    ComPtr<IWICBitmapDecoder> decoder;
    HRESULT result = factory->CreateDecoderFromFilename(
        path.c_str(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnLoad,
        &decoder);
    if (FAILED(result))
    {
        throw std::runtime_error("WIC could not decode the texture");
    }
    ComPtr<IWICBitmapFrameDecode> frame;
    if (FAILED(decoder->GetFrame(0, &frame)))
    {
        throw std::runtime_error("WIC could not read texture frame zero");
    }
    UINT width = 0;
    UINT height = 0;
    if (FAILED(frame->GetSize(&width, &height)) || width == 0 || height == 0)
    {
        throw std::runtime_error("texture dimensions are invalid");
    }
    ComPtr<IWICFormatConverter> converter;
    if (FAILED(factory->CreateFormatConverter(&converter)) ||
        FAILED(converter->Initialize(
            frame.Get(),
            GUID_WICPixelFormat32bppRGBA,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0,
            WICBitmapPaletteTypeCustom)))
    {
        throw std::runtime_error("WIC could not convert the texture to straight RGBA");
    }
    const UINT rowPitch = width * 4;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(rowPitch) * height);
    if (FAILED(converter->CopyPixels(nullptr, rowPitch, static_cast<UINT>(pixels.size()), pixels.data())))
    {
        throw std::runtime_error("WIC could not copy texture pixels");
    }

    D3D11_TEXTURE2D_DESC description = {};
    description.Width = width;
    description.Height = height;
    description.MipLevels = 1;
    description.ArraySize = 1;
    description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    description.SampleDesc.Count = 1;
    description.Usage = D3D11_USAGE_IMMUTABLE;
    description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial = {};
    initial.pSysMem = pixels.data();
    initial.SysMemPitch = rowPitch;
    ComPtr<ID3D11Texture2D> texture;
    if (FAILED(device->CreateTexture2D(&description, &initial, &texture)))
    {
        throw std::runtime_error("D3D11 could not create the texture");
    }
    ComPtr<ID3D11ShaderResourceView> view;
    if (FAILED(device->CreateShaderResourceView(texture.Get(), nullptr, &view)))
    {
        throw std::runtime_error("D3D11 could not create the texture view");
    }
    return view;
}

AlphaSummary SummarizeAlpha(
    const std::vector<std::uint8_t>& rgba,
    const unsigned int width,
    const unsigned int height)
{
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

void WriteBytes(const fs::path& path, const std::vector<std::uint8_t>& bytes)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream || !stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()))
    {
        throw std::runtime_error("failed to write raw RGBA evidence");
    }
}

void WriteReport(
    const fs::path& path,
    const unsigned int width,
    const unsigned int height,
    const AlphaSummary& summary,
    const std::map<std::string, float>& parameterValues)
{
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
    stream << ",\"driver_type\":\"d3d11-warp\",\"height\":" << height
           << ",\"nonzero_alpha_pixels\":" << summary.Nonzero
           << ",\"parameter_values\":{";
    bool firstParameter = true;
    stream << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto& parameter : parameterValues)
    {
        if (!firstParameter)
        {
            stream << ',';
        }
        firstParameter = false;
        stream << '\"' << parameter.first << "\":" << parameter.second;
    }
    stream << '}'
           << ",\"premultiplied_alpha_input\":false"
           << ",\"schema_version\":\"auto-rig-live2d-render-v1\",\"width\":" << width << '}';
    if (!stream)
    {
        throw std::runtime_error("failed to write render report");
    }
}

RenderResult Render(const Options& options)
{
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_9_1;
    const D3D_FEATURE_LEVEL requestedLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    HRESULT result = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr,
        0,
        requestedLevels,
        static_cast<UINT>(std::size(requestedLevels)),
        D3D11_SDK_VERSION,
        &device,
        &featureLevel,
        &context);
    if (result == E_INVALIDARG)
    {
        result = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_WARP,
            nullptr,
            0,
            requestedLevels + 1,
            static_cast<UINT>(std::size(requestedLevels) - 1),
            D3D11_SDK_VERSION,
            &device,
            &featureLevel,
            &context);
    }
    if (FAILED(result))
    {
        throw std::runtime_error("D3D11 WARP device creation failed");
    }

    D3D11_TEXTURE2D_DESC targetDescription = {};
    targetDescription.Width = options.Width;
    targetDescription.Height = options.Height;
    targetDescription.MipLevels = 1;
    targetDescription.ArraySize = 1;
    targetDescription.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    targetDescription.SampleDesc.Count = 1;
    targetDescription.Usage = D3D11_USAGE_DEFAULT;
    targetDescription.BindFlags = D3D11_BIND_RENDER_TARGET;
    ComPtr<ID3D11Texture2D> target;
    ComPtr<ID3D11RenderTargetView> targetView;
    if (FAILED(device->CreateTexture2D(&targetDescription, nullptr, &target)) ||
        FAILED(device->CreateRenderTargetView(target.Get(), nullptr, &targetView)))
    {
        throw std::runtime_error("D3D11 render target creation failed");
    }
    D3D11_TEXTURE2D_DESC stagingDescription = targetDescription;
    stagingDescription.Usage = D3D11_USAGE_STAGING;
    stagingDescription.BindFlags = 0;
    stagingDescription.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&stagingDescription, nullptr, &staging)))
    {
        throw std::runtime_error("D3D11 staging target creation failed");
    }

    ComPtr<IWICImagingFactory> imagingFactory;
    if (FAILED(CoCreateInstance(
            CLSID_WICImagingFactory,
            nullptr,
            CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&imagingFactory))))
    {
        throw std::runtime_error("WIC factory creation failed");
    }
    ComPtr<ID3D11ShaderResourceView> textureView =
        LoadTexture(device.Get(), imagingFactory.Get(), options.TexturePath);

    FrameworkOptions.LogFunction = CoreLog;
    FrameworkOptions.LoggingLevel = Csm::CubismFramework::Option::LogLevel_Info;
    FrameworkOptions.LoadFileFunction = LoadFrameworkFile;
    FrameworkOptions.ReleaseBytesFunction = ReleaseFrameworkFile;
    if (!Csm::CubismFramework::StartUp(&FrameworkAllocator, &FrameworkOptions))
    {
        throw std::runtime_error("Cubism Framework startup failed");
    }
    Csm::CubismFramework::Initialize();
    Csm::Rendering::CubismRenderer_D3D11::SetConstantSettings(1, device.Get());

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
        motion->SetFadeInTime(0.0f);
        motion->SetFadeOutTime(0.0f);
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
    std::map<std::string, float> observedParameterValues;
    for (const std::string& parameterId : options.ObservedParameters)
    {
        const Csm::CubismIdHandle id = Csm::CubismFramework::GetIdManager()->GetId(parameterId.c_str());
        observedParameterValues.emplace(parameterId, model->GetParameterValue(id));
    }

    Csm::Rendering::CubismRenderer* rendererBase =
        Csm::Rendering::CubismRenderer::Create(options.Width, options.Height);
    if (rendererBase == nullptr)
    {
        moc->DeleteModel(model);
        Csm::CubismMoc::Delete(moc);
        Csm::CubismFramework::Dispose();
        throw std::runtime_error("Cubism Framework could not create the D3D11 renderer");
    }
    auto* renderer = static_cast<Csm::Rendering::CubismRenderer_D3D11*>(rendererBase);
    renderer->Initialize(model);
    renderer->BindTexture(0, textureView.Get());
    renderer->IsPremultipliedAlpha(false);
    renderer->IsCulling(false);
    Csm::CubismMatrix44 matrix;
    matrix.LoadIdentity();
    renderer->SetMvpMatrix(&matrix);

    ID3D11RenderTargetView* targetViews[] = {targetView.Get()};
    context->OMSetRenderTargets(1, targetViews, nullptr);
    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(options.Width);
    viewport.Height = static_cast<float>(options.Height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    context->RSSetViewports(1, &viewport);
    const float clearColor[] = {0.0f, 0.0f, 0.0f, 0.0f};
    context->ClearRenderTargetView(targetView.Get(), clearColor);
    renderer->StartFrame(context.Get());
    renderer->DrawModel();
    renderer->EndFrame();
    context->CopyResource(staging.Get(), target.Get());

    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped)))
    {
        Csm::Rendering::CubismRenderer::Delete(renderer);
        moc->DeleteModel(model);
        Csm::CubismMoc::Delete(moc);
        Csm::Rendering::CubismDeviceInfo_D3D11::ReleaseDeviceInfo(device.Get());
        Csm::Rendering::CubismRenderer::StaticRelease();
        Csm::CubismFramework::Dispose();
        throw std::runtime_error("D3D11 staging target could not be mapped");
    }
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(options.Width) * options.Height * 4);
    for (unsigned int row = 0; row < options.Height; ++row)
    {
        const auto* source = static_cast<const std::uint8_t*>(mapped.pData) + mapped.RowPitch * row;
        std::copy(source, source + options.Width * 4, rgba.begin() + options.Width * 4 * row);
    }
    context->Unmap(staging.Get(), 0);

    Csm::Rendering::CubismRenderer::Delete(renderer);
    moc->DeleteModel(model);
    Csm::CubismMoc::Delete(moc);
    Csm::Rendering::CubismDeviceInfo_D3D11::ReleaseDeviceInfo(device.Get());
    Csm::Rendering::CubismRenderer::StaticRelease();
    context->ClearState();
    context->Flush();
    textureView.Reset();
    imagingFactory.Reset();
    staging.Reset();
    targetView.Reset();
    target.Reset();
    context.Reset();
    device.Reset();
    Csm::CubismFramework::Dispose();
    mocBytes.clear();
    mocBytes.shrink_to_fit();
    return RenderResult{std::move(rgba), std::move(observedParameterValues)};
}
} // namespace

int main(int argc, char** argv)
{
    const HRESULT initializeResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool shouldUninitialize = SUCCEEDED(initializeResult);
    try
    {
        const Options options = ParseOptions(argc, argv);
        const RenderResult result = Render(options);
        const AlphaSummary summary = SummarizeAlpha(result.Rgba, options.Width, options.Height);
        WriteBytes(options.OutputPath, result.Rgba);
        WriteReport(
            options.ReportPath,
            options.Width,
            options.Height,
            summary,
            result.ParameterValues);
        if (shouldUninitialize)
        {
            CoUninitialize();
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        if (shouldUninitialize)
        {
            CoUninitialize();
        }
        return 1;
    }
}
