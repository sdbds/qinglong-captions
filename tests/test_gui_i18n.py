from gui.utils.i18n import I18n, TRANSLATIONS


def _flatten_keys(mapping: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.update(_flatten_keys(value, full_key))
        else:
            keys.add(full_key)
    return keys


def test_gui_i18n_languages_have_same_leaf_keys():
    expected = _flatten_keys(TRANSLATIONS["en"])

    for lang, mapping in TRANSLATIONS.items():
        assert _flatten_keys(mapping) == expected, lang


def test_recent_gui_i18n_keys_are_available_in_all_languages():
    keys = [
        "job_list_title",
        "job_status_pending",
        "path_copied",
        "log_copied",
        "codex_subscription",
        "use_codex_subscription",
        "codex_fast_mode",
        "codex_reasoning_effort",
        "kimi_code_thinking",
        "grok_build_reasoning_effort",
        "grok_build_disable_web_search",
        "repo_id_layerdiff",
        "repo_id_depth",
        "search_or_select",
        "items_count",
        "music_transcription",
        "music_transcription_preview",
        "job_name_music_transcription",
    ]

    for lang in TRANSLATIONS:
        i18n = I18n(lang)
        for key in keys:
            assert i18n.t(key) != key


def test_removed_grok_build_effort_key_is_not_translated():
    for mapping in TRANSLATIONS.values():
        assert "grok_build_effort" not in _flatten_keys(mapping)
