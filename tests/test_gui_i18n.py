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
        "repo_id_layerdiff",
        "repo_id_depth",
        "search_or_select",
        "items_count",
    ]

    for lang in TRANSLATIONS:
        i18n = I18n(lang)
        for key in keys:
            assert i18n.t(key) != key
