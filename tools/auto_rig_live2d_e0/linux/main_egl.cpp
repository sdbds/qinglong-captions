#include "validator_common.hpp"

#include <GL/glew.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <Math/CubismMatrix44.hpp>
#include <Rendering/CubismRenderer.hpp>
#include <Rendering/OpenGL/CubismRenderer_OpenGLES2.hpp>

#define STBI_NO_STDIO
#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace qinglong::live2d
{
namespace
{
std::string EglError(const char* operation)
{
    std::ostringstream message;
    message << operation << " failed with EGL error 0x" << std::hex << std::uppercase << eglGetError();
    return message.str();
}

bool HasExtension(const char* extensions, const std::string& expected)
{
    if (extensions == nullptr || expected.empty() || expected.find(' ') != std::string::npos)
    {
        return false;
    }
    const std::string values(extensions);
    std::size_t offset = 0;
    while ((offset = values.find(expected, offset)) != std::string::npos)
    {
        const bool startsAtBoundary = offset == 0 || values[offset - 1] == ' ';
        const std::size_t end = offset + expected.size();
        const bool endsAtBoundary = end == values.size() || values[end] == ' ';
        if (startsAtBoundary && endsAtBoundary)
        {
            return true;
        }
        offset = end;
    }
    return false;
}

class EglContext final
{
public:
    EglContext(const unsigned int width, const unsigned int height)
    {
        try
        {
            const char* clientExtensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
            if (!HasExtension(clientExtensions, "EGL_MESA_platform_surfaceless"))
            {
                throw std::runtime_error(
                    "EGL_MESA_platform_surfaceless is required for the Linux headless validator");
            }
            const auto getPlatformDisplay = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
                eglGetProcAddress("eglGetPlatformDisplayEXT"));
            if (getPlatformDisplay == nullptr)
            {
                throw std::runtime_error("eglGetPlatformDisplayEXT is unavailable");
            }
            _display = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
            if (_display == EGL_NO_DISPLAY)
            {
                throw std::runtime_error(EglError("eglGetPlatformDisplayEXT(surfaceless)"));
            }
            if (eglInitialize(_display, &_majorVersion, &_minorVersion) != EGL_TRUE)
            {
                throw std::runtime_error(EglError("eglInitialize"));
            }
            if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE)
            {
                throw std::runtime_error(EglError("eglBindAPI(OpenGL)"));
            }

            const EGLint configAttributes[] = {
                EGL_SURFACE_TYPE,
                EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE,
                EGL_OPENGL_BIT,
                EGL_RED_SIZE,
                8,
                EGL_GREEN_SIZE,
                8,
                EGL_BLUE_SIZE,
                8,
                EGL_ALPHA_SIZE,
                8,
                EGL_NONE,
            };
            EGLConfig config = nullptr;
            EGLint configCount = 0;
            if (eglChooseConfig(_display, configAttributes, &config, 1, &configCount) != EGL_TRUE ||
                configCount != 1)
            {
                throw std::runtime_error(EglError("eglChooseConfig(RGBA8 pbuffer)"));
            }

            const EGLint surfaceAttributes[] = {
                EGL_WIDTH,
                static_cast<EGLint>(width),
                EGL_HEIGHT,
                static_cast<EGLint>(height),
                EGL_NONE,
            };
            _surface = eglCreatePbufferSurface(_display, config, surfaceAttributes);
            if (_surface == EGL_NO_SURFACE)
            {
                throw std::runtime_error(EglError("eglCreatePbufferSurface"));
            }
            const EGLint contextAttributes[] = {EGL_NONE};
            _context = eglCreateContext(_display, config, EGL_NO_CONTEXT, contextAttributes);
            if (_context == EGL_NO_CONTEXT)
            {
                throw std::runtime_error(EglError("eglCreateContext"));
            }
            if (eglMakeCurrent(_display, _surface, _surface, _context) != EGL_TRUE)
            {
                throw std::runtime_error(EglError("eglMakeCurrent"));
            }

            glewExperimental = GL_TRUE;
            const GLenum glewResult = glewInit();
            // GLEW may leave GL_INVALID_ENUM after probing legacy entry points.
            glGetError();
            if (glewResult != GLEW_OK)
            {
                const char* detail = reinterpret_cast<const char*>(glewGetErrorString(glewResult));
                throw std::runtime_error(
                    std::string("GLEW initialization failed: ") +
                    (detail == nullptr ? "unknown" : detail));
            }
        }
        catch (...)
        {
            Reset();
            throw;
        }
    }

    ~EglContext()
    {
        Reset();
    }

    EglContext(const EglContext&) = delete;
    EglContext& operator=(const EglContext&) = delete;

    EGLDisplay Display() const noexcept
    {
        return _display;
    }

    std::string Version() const
    {
        return std::to_string(_majorVersion) + "." + std::to_string(_minorVersion);
    }

private:
    void Reset() noexcept
    {
        if (_display == EGL_NO_DISPLAY)
        {
            return;
        }
        eglMakeCurrent(_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (_context != EGL_NO_CONTEXT)
        {
            eglDestroyContext(_display, _context);
            _context = EGL_NO_CONTEXT;
        }
        if (_surface != EGL_NO_SURFACE)
        {
            eglDestroySurface(_display, _surface);
            _surface = EGL_NO_SURFACE;
        }
        eglTerminate(_display);
        _display = EGL_NO_DISPLAY;
    }

    EGLDisplay _display = EGL_NO_DISPLAY;
    EGLSurface _surface = EGL_NO_SURFACE;
    EGLContext _context = EGL_NO_CONTEXT;
    EGLint _majorVersion = 0;
    EGLint _minorVersion = 0;
};

std::string GlString(const GLenum name)
{
    const GLubyte* value = glGetString(name);
    return value == nullptr ? "unavailable" : reinterpret_cast<const char*>(value);
}

std::string EglString(const EGLDisplay display, const EGLint name)
{
    const char* value = eglQueryString(display, name);
    return value == nullptr ? "unavailable" : value;
}

GLuint LoadTexture(const fs::path& path)
{
    const std::vector<std::uint8_t> encoded = ReadFile(path);
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(
        encoded.data(),
        static_cast<int>(encoded.size()),
        &width,
        &height,
        &channels,
        STBI_rgb_alpha);
    if (pixels == nullptr || width <= 0 || height <= 0)
    {
        stbi_image_free(pixels);
        throw std::runtime_error("stb_image could not decode the PNG texture");
    }

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_image_free(pixels);
    if (texture == 0 || glGetError() != GL_NO_ERROR)
    {
        throw std::runtime_error("OpenGL texture upload failed");
    }
    return texture;
}

class EglHeadlessBackend final : public ValidatorBackend
{
public:
    const char* BackendId() const noexcept override
    {
        return "opengl-egl-headless";
    }

    BackendRenderResult Render(const Options& options, Csm::CubismModel* model) override
    {
        EglContext context(options.Width, options.Height);
        std::vector<GLuint> textures;
        textures.reserve(options.TexturePaths.size());
        for (const fs::path& texturePath : options.TexturePaths)
        {
            textures.push_back(LoadTexture(texturePath));
        }

        GLuint framebuffer = 0;
        GLuint colorTexture = 0;
        glGenFramebuffers(1, &framebuffer);
        glGenTextures(1, &colorTexture);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            static_cast<GLsizei>(options.Width),
            static_cast<GLsizei>(options.Height),
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            throw std::runtime_error("OpenGL RGBA8 framebuffer is incomplete");
        }

        Csm::Rendering::CubismRenderer* rendererBase =
            Csm::Rendering::CubismRenderer::Create(options.Width, options.Height);
        if (rendererBase == nullptr)
        {
            throw std::runtime_error("Cubism Framework could not create the OpenGL renderer");
        }
        auto* renderer = static_cast<Csm::Rendering::CubismRenderer_OpenGLES2*>(rendererBase);
        renderer->Initialize(model);
        for (std::size_t textureIndex = 0; textureIndex < textures.size(); ++textureIndex)
        {
            renderer->BindTexture(static_cast<Csm::csmUint32>(textureIndex), textures[textureIndex]);
        }
        renderer->IsPremultipliedAlpha(false);
        renderer->IsCulling(false);
        Csm::CubismMatrix44 matrix;
        matrix.LoadIdentity();
        renderer->SetMvpMatrix(&matrix);

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glViewport(0, 0, static_cast<GLsizei>(options.Width), static_cast<GLsizei>(options.Height));
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_STENCIL_TEST);
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        renderer->DrawModel();
        glFinish();

        std::vector<std::uint8_t> bottomUp(static_cast<std::size_t>(options.Width) * options.Height * 4);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(
            0,
            0,
            static_cast<GLsizei>(options.Width),
            static_cast<GLsizei>(options.Height),
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            bottomUp.data());
        if (glGetError() != GL_NO_ERROR)
        {
            Csm::Rendering::CubismRenderer::Delete(renderer);
            throw std::runtime_error("OpenGL framebuffer readback failed");
        }
        std::vector<std::uint8_t> rgba(bottomUp.size());
        const std::size_t rowBytes = static_cast<std::size_t>(options.Width) * 4;
        for (unsigned int row = 0; row < options.Height; ++row)
        {
            const std::size_t sourceRow = options.Height - 1 - row;
            std::copy_n(bottomUp.data() + sourceRow * rowBytes, rowBytes, rgba.data() + row * rowBytes);
        }

        const std::map<std::string, std::string> runtimeInfo = {
            {"api", "opengl-egl"},
            {"egl_client_apis", EglString(context.Display(), EGL_CLIENT_APIS)},
            {"egl_vendor", EglString(context.Display(), EGL_VENDOR)},
            {"egl_version", context.Version()},
            {"renderer", GlString(GL_RENDERER)},
            {"vendor", GlString(GL_VENDOR)},
            {"version", GlString(GL_VERSION)},
        };
        Csm::Rendering::CubismRenderer::Delete(renderer);
        Csm::Rendering::CubismRenderer::StaticRelease();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &framebuffer);
        glDeleteTextures(1, &colorTexture);
        glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
        return BackendRenderResult{std::move(rgba), runtimeInfo};
    }
};
} // namespace
} // namespace qinglong::live2d

int main(int argc, char** argv)
{
    qinglong::live2d::EglHeadlessBackend backend;
    return qinglong::live2d::RunValidator(argc, argv, backend);
}
