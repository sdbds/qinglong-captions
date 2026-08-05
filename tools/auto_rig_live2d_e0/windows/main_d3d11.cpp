#include "validator_common.hpp"

#include <Windows.h>
#include <d3d11.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <Math/CubismMatrix44.hpp>
#include <Rendering/CubismRenderer.hpp>
#include <Rendering/D3D11/CubismDeviceInfo_D3D11.hpp>
#include <Rendering/D3D11/CubismRenderer_D3D11.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

namespace qinglong::live2d
{
namespace
{
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

const char* FeatureLevelName(const D3D_FEATURE_LEVEL level)
{
    switch (level)
    {
    case D3D_FEATURE_LEVEL_11_1:
        return "11.1";
    case D3D_FEATURE_LEVEL_11_0:
        return "11.0";
    case D3D_FEATURE_LEVEL_10_1:
        return "10.1";
    case D3D_FEATURE_LEVEL_10_0:
        return "10.0";
    default:
        return "unknown";
    }
}

class D3D11WarpBackend final : public ValidatorBackend
{
public:
    const char* BackendId() const noexcept override
    {
        return "d3d11-warp";
    }

    BackendRenderResult Render(const Options& options, Csm::CubismModel* model) override
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
        std::vector<ComPtr<ID3D11ShaderResourceView>> textureViews;
        textureViews.reserve(options.TexturePaths.size());
        for (const fs::path& texturePath : options.TexturePaths)
        {
            textureViews.push_back(LoadTexture(device.Get(), imagingFactory.Get(), texturePath));
        }

        Csm::Rendering::CubismRenderer_D3D11::SetConstantSettings(1, device.Get());
        Csm::Rendering::CubismRenderer* rendererBase =
            Csm::Rendering::CubismRenderer::Create(options.Width, options.Height);
        if (rendererBase == nullptr)
        {
            throw std::runtime_error("Cubism Framework could not create the D3D11 renderer");
        }
        auto* renderer = static_cast<Csm::Rendering::CubismRenderer_D3D11*>(rendererBase);
        renderer->Initialize(model);
        for (std::size_t textureIndex = 0; textureIndex < textureViews.size(); ++textureIndex)
        {
            renderer->BindTexture(static_cast<Csm::csmUint32>(textureIndex), textureViews[textureIndex].Get());
        }
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
        Csm::Rendering::CubismDeviceInfo_D3D11::ReleaseDeviceInfo(device.Get());
        Csm::Rendering::CubismRenderer::StaticRelease();
        context->ClearState();
        context->Flush();
        return BackendRenderResult{
            std::move(rgba),
            {
                {"api", "d3d11"},
                {"driver", "warp"},
                {"feature_level", FeatureLevelName(featureLevel)},
            },
        };
    }
};
} // namespace
} // namespace qinglong::live2d

int main(int argc, char** argv)
{
    const HRESULT initializeResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool shouldUninitialize = SUCCEEDED(initializeResult);
    qinglong::live2d::D3D11WarpBackend backend;
    const int result = qinglong::live2d::RunValidator(argc, argv, backend);
    if (shouldUninitialize)
    {
        CoUninitialize();
    }
    return result;
}
