from tests.provider_v2_helpers import make_provider_args


class TestFindProvider:
    def test_stepfun(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(step_api_key="sk-xxx")
        assert reg.find_provider(args, "image/jpeg").name == "stepfun"
        assert reg.find_provider(args, "video/mp4").name == "stepfun"

    def test_stepfun_does_not_claim_pdf(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(step_api_key="sk-xxx")
        provider = reg.find_provider(args, "application/pdf")
        assert provider is None

    def test_ark_video(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ark_api_key="ak-xxx")
        provider = reg.find_provider(args, "video/mp4")
        assert provider is not None and provider.name == "ark"

    def test_qwenvl_video(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(qwenVL_api_key="qk-xxx")
        provider = reg.find_provider(args, "video/mp4")
        assert provider is not None and provider.name == "qwenvl"

    def test_glm_video(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(glm_api_key="gk-xxx")
        provider = reg.find_provider(args, "video/mp4")
        assert provider is not None and provider.name == "glm"

    def test_kimi_code_priority_over_kimi_vl(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(kimi_code_api_key="kc-xxx", kimi_api_key="kv-xxx")
        provider = reg.find_provider(args, "image/jpeg")
        assert provider.name == "kimi_code"

    def test_kimi_vl_when_no_kimi_code(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(kimi_api_key="kv-xxx")
        provider = reg.find_provider(args, "image/jpeg")
        assert provider.name == "kimi_vl"

    def test_gemini(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(gemini_api_key="gm-xxx")
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is not None and provider.name == "gemini"

    def test_mistral_image(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(pixtral_api_key="px-xxx")
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is not None and provider.name == "mistral_ocr"

    def test_mistral_pdf(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(pixtral_api_key="px-xxx")
        provider = reg.find_provider(args, "application/pdf")
        assert provider is not None and provider.name == "mistral_ocr"

    def test_mistral_ocr_alias_mode(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ocr_model="pixtral_ocr", pixtral_api_key="pk-xxx", document_image=True)
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is not None and provider.name == "mistral_ocr"

    def test_mistral_ocr_mode_pdf(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ocr_model="pixtral_ocr", pixtral_api_key="pk-xxx")
        provider = reg.find_provider(args, "application/pdf")
        assert provider is not None and provider.name == "mistral_ocr"

    def test_mistral_ocr_canonical_mode(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ocr_model="mistral_ocr", mistral_api_key="mk-xxx", document_image=True)
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is not None and provider.name == "mistral_ocr"

    def test_deepseek_ocr(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ocr_model="deepseek_ocr", document_image=True)
        provider = reg.find_provider(args, "image/png")
        assert provider is not None and provider.name == "deepseek_ocr"

    def test_logics_ocr(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ocr_model="logics_ocr", document_image=True)
        provider = reg.find_provider(args, "image/png")
        assert provider is not None and provider.name == "logics_ocr"

    def test_lighton_ocr(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ocr_model="lighton_ocr", document_image=True)
        provider = reg.find_provider(args, "image/png")
        assert provider is not None and provider.name == "lighton_ocr"

    def test_dots_ocr(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ocr_model="dots_ocr", document_image=True)
        provider = reg.find_provider(args, "image/png")
        assert provider is not None and provider.name == "dots_ocr"

    def test_qianfan_ocr(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ocr_model="qianfan_ocr", document_image=True)
        provider = reg.find_provider(args, "image/png")
        assert provider is not None and provider.name == "qianfan_ocr"

    def test_explicit_ocr_route_beats_cloud_priority_for_images(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(step_api_key="sk-xxx", ocr_model="paddle_ocr", document_image=True)
        provider = reg.find_provider(args, "image/png")
        assert provider is not None and provider.name == "paddle_ocr"

    def test_explicit_ocr_route_beats_cloud_priority_for_pdfs(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(step_api_key="sk-xxx", ocr_model="paddle_ocr")
        provider = reg.find_provider(args, "application/pdf")
        assert provider is not None and provider.name == "paddle_ocr"

    def test_kimi_code_does_not_steal_explicit_ocr_route(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(kimi_code_api_key="kc-xxx", ocr_model="qianfan_ocr", document_image=True)
        provider = reg.find_provider(args, "image/png")
        assert provider is not None and provider.name == "qianfan_ocr"

    def test_ocr_pdf_always_handled(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ocr_model="hunyuan_ocr")
        provider = reg.find_provider(args, "application/pdf")
        assert provider is not None and provider.name == "hunyuan_ocr"

    def test_ocr_image_requires_document_image(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(ocr_model="glm_ocr", document_image=False)
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is None or provider.name != "glm_ocr"

    def test_moondream_vlm(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(vlm_image_model="moondream")
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is not None and provider.name == "moondream"

    def test_vlm_route_ignores_irrelevant_ocr_setting(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(vlm_image_model="moondream", ocr_model="paddle_ocr", document_image=False)
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is not None and provider.name == "moondream"

    def test_qwen_vl_local(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(vlm_image_model="qwen_vl_local")
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is not None and provider.name == "qwen_vl_local"

    def test_step_vl_local(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(vlm_image_model="step_vl_local")
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is not None and provider.name == "step_vl_local"

    def test_reka_edge_local_image(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(vlm_image_model="reka_edge_local")
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is not None and provider.name == "reka_edge_local"

    def test_reka_edge_local_video(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(vlm_image_model="reka_edge_local")
        provider = reg.find_provider(args, "video/mp4")
        assert provider is not None and provider.name == "reka_edge_local"

    def test_lfm_vl_local_image(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(vlm_image_model="lfm_vl_local")
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is not None and provider.name == "lfm_vl_local"

    def test_music_flamingo_local_audio(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(alm_model="music_flamingo_local")
        provider = reg.find_provider(args, "audio/wav")
        assert provider is not None and provider.name == "music_flamingo_local"

    def test_music_flamingo_local_ignores_non_audio(self):
        from providers.registry import get_registry

        reg = get_registry()
        args = make_provider_args(alm_model="music_flamingo_local")
        provider = reg.find_provider(args, "image/jpeg")
        assert provider is None or provider.name != "music_flamingo_local"

    def test_all_ocr_providers(self):
        from providers.registry import get_registry

        reg = get_registry()
        for name in (
            "deepseek_ocr",
            "logics_ocr",
            "dots_ocr",
            "qianfan_ocr",
            "lighton_ocr",
            "hunyuan_ocr",
            "glm_ocr",
            "chandra_ocr",
            "olmocr",
            "paddle_ocr",
            "nanonets_ocr",
            "firered_ocr",
        ):
            provider = reg.find_provider(make_provider_args(ocr_model=name), "application/pdf")
            assert provider is not None and provider.name == name, f"OCR provider {name} not found for PDF"
