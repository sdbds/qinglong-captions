"""
青龙字幕工具 GUI (Qinglong Captions GUI)
基于 NiceGUI 的现代化图形化界面

使用方法:
    cd gui
    python main.py

或者:
    python gui/main.py
"""

import sys
from importlib import import_module
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncio
import logging
import toml
from nicegui import ui, app
from theme import apply_theme, get_classes, COLORS
from gui.utils.i18n import t, set_language, get_i18n
from components.job_list import JobListDrawer, inject_job_list_css


def _load_app_version() -> str:
    """Read GUI version from the project metadata instead of duplicating constants."""
    try:
        pyproject = toml.load(project_root / "pyproject.toml")
        return str(pyproject.get("project", {}).get("version", "")).strip() or "0.0.0"
    except Exception:
        return "0.0.0"


# 页面标题和样式
APP_TITLE = "青龙字幕工具"
APP_TITLE_EN = "Qinglong Captions"
APP_VERSION = _load_app_version()

# 全局主题脚本
THEME_SCRIPT = """
<script>
(function() {
    function syncQuasarDark(isDark) {
        if (window.Quasar && Quasar.Dark) Quasar.Dark.set(isDark);
        var accent = isDark ? '#daa520' : '#c49318';
        document.documentElement.style.setProperty('--q-primary', accent);
        document.documentElement.style.setProperty('--q-color-primary', accent);
    }

    window.toggleDarkMode = function() {
        var isDark = document.body.classList.toggle('dark-mode');
        localStorage.setItem('dark_mode', isDark);
        syncQuasarDark(isDark);
        document.querySelectorAll('.theme-toggle-btn .q-icon').forEach(function(icon) {
            icon.textContent = isDark ? 'dark_mode' : 'light_mode';
        });
        return isDark;
    };

    var saved = localStorage.getItem('dark_mode');
    var isDark = saved === null ? true : saved === 'true';
    if (isDark) document.body.classList.add('dark-mode');
    else document.body.classList.remove('dark-mode');
    syncQuasarDark(isDark);

    document.querySelectorAll('.theme-toggle-btn .q-icon').forEach(function(icon) {
        icon.textContent = isDark ? 'dark_mode' : 'light_mode';
    });

    setTimeout(function() { syncQuasarDark(isDark); }, 300);
})();
</script>
"""


def _load_wizard_attr(module_name: str, attr_name: str):
    """Load wizard modules on demand to keep startup on the homepage cheap."""
    module = import_module(module_name)
    return getattr(module, attr_name)


def create_header(job_drawer=None):
    """创建现代化页面头部导航"""
    with ui.header().classes(get_classes("header")):
        with ui.row().classes("w-full items-center justify-between").style("padding: 4px 0;"):
            # Logo
            with ui.row().classes("items-center gap-3"):
                ui.label("🐉").style(
                    "font-size: 28px; line-height: 1;"
                )
                with ui.column().classes("gap-0"):
                    ui.label(t("app_title")).classes("text-subtitle1").classes("header-title")
                    ui.label(f"v{APP_VERSION}").classes("header-version")

            # Nav
            with ui.row().classes("gap-1"):
                nav_items = [
                    (t("nav_home"), "/", "home"),
                    (t("nav_import"), "/import", "download"),
                    (t("nav_split"), "/split", "movie"),
                    (t("nav_tagger"), "/tagger", "label"),
                    (t("nav_caption"), "/caption", "subtitles"),
                    (t("nav_export"), "/export", "upload"),
                    (t("nav_tools"), "/tools", "construction"),
                ]

                for label_text, path, icon in nav_items:
                    btn = ui.button(label_text, on_click=lambda p=path: ui.navigate.to(p), icon=icon, color=None)
                    btn.props("flat no-caps")
                    btn.classes(get_classes("nav_btn"))

            # Right side: Job list + Settings + Theme + Language
            with ui.row().classes("items-center gap-2"):
                if job_drawer is not None:
                    job_btn = ui.button(icon="assignment", on_click=job_drawer.toggle).props("flat round dense")
                    job_btn.style("color: var(--ql-text-secondary);")
                    job_btn.tooltip("任务列表")
                    job_badge = ui.badge("0", color="red").props("floating").style("display: none;")
                    job_drawer.set_badge(job_badge)

                settings_dlg = _load_wizard_attr("wizard.step7_settings", "create_settings_dialog")()
                settings_btn = ui.button(icon="settings", on_click=settings_dlg.open).props("flat round dense")
                settings_btn.style("color: var(--ql-text-secondary);")
                settings_btn.tooltip(t("nav_settings"))

                theme_btn = ui.button(icon="light_mode").props("flat round dense").classes("theme-toggle-btn")
                theme_btn.on_click(lambda: ui.run_javascript("window.toggleDarkMode()"))

                with ui.row().classes("items-center gap-1"):
                    ui.icon("language", size="18px").style("color: var(--ql-text-muted);")
                    lang_select = (
                        ui.select(
                            {
                                "zh": "🇨🇳 中文",
                                "en": "🇺🇸 English",
                                "ja": "🇯🇵 日本語",
                                "ko": "🇰🇷 한국어",
                            },
                            label="",
                            value=get_i18n().lang,
                        )
                        .props('dense outlined use-input fill-input hide-selected input-debounce="0" dropdown-icon="expand_more"')
                        .classes("lang-selector")
                    )

                    def on_lang_change(e):
                        lang = e.value
                        if lang and lang in ["zh", "en", "ja", "ko"]:
                            set_language(lang)
                            ui.notify(t("language_changed"), type="positive")
                            ui.run_javascript("window.location.reload();")

                    lang_select.on_value_change(on_lang_change)


def page_base(content_func):
    """页面基础包装器 - 应用主题和脚本"""
    # 应用主题
    apply_theme()

    # 添加主题脚本（只执行一次）
    ui.add_body_html(THEME_SCRIPT)

    # Job 列表抽屉（top-level layout element，必须是页面直接子元素）
    inject_job_list_css()
    job_drawer = JobListDrawer()

    # 创建头部（传入 drawer 供按钮绑定）
    create_header(job_drawer)

    # 执行页面内容
    content_func()


def home_page():
    """首页/欢迎页面"""

    def content():
        with ui.column().classes(get_classes("page_container") + " gap-6"):
            # Hero Section
            with ui.element("div").classes("w-full text-center").style("padding: 48px 0 32px;"):
                ui.label(t("app_title")).classes("text-h3").style(
                    "font-weight: 600; "
                    "background: linear-gradient(135deg, var(--ql-accent), var(--ql-secondary)); "
                    "-webkit-background-clip: text; -webkit-text-fill-color: transparent; "
                    "background-clip: text;"
                )
                ui.label(t("app_description")).classes("text-body1 app-desc").style("margin-top: 8px;")

            # Quick Start Cards
            with ui.card().classes(get_classes("card") + " w-full q-pa-lg"):
                with ui.row().classes("w-full items-center justify-between q-mb-md"):
                    ui.label(t("quick_start")).classes("text-h6 section-title").style("font-weight: 600;")
                    with ui.row().classes("gap-2 items-center"):
                        ui.label(t("support")).classes("text-caption").style("color: var(--ql-text-muted);")
                        ui.label("6").classes(get_classes("badge") + " ql-badge--primary")
                        ui.label(t("tools_label")).classes("text-caption").style("color: var(--ql-text-muted);")

                with ui.row().classes("w-full gap-3"):
                    steps = [
                        ("download", t("dataset_import"), t("feature_list")["import_desc"], "/import"),
                        ("movie", t("video_split"), t("feature_list")["split_desc"], "/split"),
                        ("label", t("dataset_tagging"), t("feature_list")["tagger_desc"], "/tagger"),
                        ("subtitles", t("caption_generate"), t("feature_list")["caption_desc"], "/caption"),
                        ("upload", t("dataset_export"), t("feature_list")["export_desc"], "/export"),
                        ("construction", t("tools"), t("feature_list")["tools_desc"], "/tools"),
                    ]

                    for icon, title, desc, path in steps:
                        with ui.card().classes("step-card flex-1").on("click", lambda p=path: ui.navigate.to(p)):
                            ui.icon(icon, size="36px").style("margin-bottom: 10px;")
                            ui.label(title).classes("text-subtitle1 step-title").style("font-weight: 600;")
                            ui.label(desc).classes("text-caption step-desc")

            # Supported Models
            with ui.card().classes(get_classes("card") + " w-full q-pa-lg"):
                with ui.row().classes("w-full items-center gap-3 q-mb-md"):
                    ui.icon("view_module", size="24px").style("color: var(--ql-accent);")
                    ui.label(t("supported_models")).classes("text-h6 section-title").style("font-weight: 600;")

                model_list = t("model_list")
                models = [
                    ("Gemini", model_list["gemini"]),
                    ("Mistral", model_list["pixtral"]),
                    ("Step-VL", model_list["step"]),
                    ("Qwen-VL", model_list["qwen"]),
                    ("Kimi", model_list["kimi"]),
                    ("Kimi-Code", model_list["kimi_code"]),
                    ("MiniMax", model_list["minimax"]),
                    ("MiniMax-Code", model_list["minimax_code"]),
                    ("GLM", model_list["glm"]),
                    ("Ark", model_list["ark"]),
                    ("OpenAI-Compatible", model_list["openai_compatible"]),
                ]

                with ui.grid(columns=3).classes("w-full gap-3"):
                    for name, desc in models:
                        with ui.row().classes("model-item items-center gap-3"):
                            ui.icon("check_circle", size="18px").style("color: var(--ql-accent);")
                            with ui.column().classes("gap-0"):
                                ui.label(name).classes("text-body2 model-name").style("font-weight: 500;")
                                ui.label(desc).classes("text-caption model-desc")

            # Features Section
            with ui.card().classes(get_classes("card") + " w-full q-pa-lg"):
                with ui.row().classes("w-full items-center gap-3 q-mb-md"):
                    ui.icon("auto_awesome", size="24px").style("color: var(--ql-secondary);")
                    ui.label(t("features")).classes("text-h6 section-title").style("font-weight: 600;")

                feature_list = t("feature_list")
                features = [
                    ("🎬", feature_list["video_caption"]),
                    ("🏷️", feature_list["auto_tagging"]),
                    ("🤖", feature_list["multi_api"]),
                    ("💾", feature_list["lance_db"]),
                    ("📊", feature_list["batch_process"]),
                    ("🎨", feature_list["modern_ui"]),
                ]

                with ui.grid(columns=3).classes("w-full gap-3"):
                    for icon, desc in features:
                        with ui.row().classes("model-item items-start gap-3"):
                            ui.label(icon).classes("text-xl")
                            with ui.column().classes("gap-0"):
                                ui.label(desc).classes("text-body2 feature-desc")

    page_base(content)


def import_page():
    """数据导入页面"""
    page_base(_load_wizard_attr("wizard.step1_import", "render_import_step"))


def split_page():
    """视频分割页面"""
    page_base(_load_wizard_attr("wizard.step2_video_split", "render_video_split_step"))


def tagger_page():
    """标签生成页面"""
    page_base(_load_wizard_attr("wizard.step3_tagger", "render_tagger_step"))


def caption_page():
    """字幕生成页面"""
    page_base(_load_wizard_attr("wizard.step4_caption", "render_caption_step"))


def export_page():
    """数据导出页面"""
    page_base(_load_wizard_attr("wizard.step5_export", "render_export_step"))


def tools_page():
    """工具页面"""
    page_base(_load_wizard_attr("wizard.step6_tools", "render_tools_step"))


def setup_page():
    """环境检查页面"""
    page_base(_load_wizard_attr("wizard.step0_setup", "render_setup_step"))


def console_page():
    """全屏终端页面（无导航栏）"""
    _load_wizard_attr("wizard.console_page", "render_console_page")()


def not_found_page():
    """404 页面 - 页面不存在"""

    def content():
        with ui.column().classes(get_classes("page_container") + " gap-6 items-center justify-center").style("min-height: 60vh;"):
            ui.icon("error_outline", size="80px").style("color: var(--ql-warning);")
            ui.label(t("page_not_found")).classes("text-h3").style("font-weight: 600; color: var(--ql-text);")
            ui.label(t("page_not_found_desc")).classes("text-body1").style("color: var(--ql-text-secondary);")

            with ui.row().classes("gap-4 q-mt-lg"):
                home_btn = ui.button(t("back_to_home"), on_click=lambda: ui.navigate.to("/"), icon="home")
                home_btn.classes("modern-btn-primary")
                home_btn.classes(remove="bg-primary bg-secondary text-white")

    page_base(content)


def _install_exception_filter():
    """monkey-patch app.handle_exception，在 PATH A 之前拦截 parent_slot 竞态错误。

    NiceGUI 的 app.handle_exception 先走 PATH A（client handler），再走 PATH B（app handlers）。
    替换 _exception_handlers 列表只能影响 PATH B，无法阻止 PATH A 的级联异常逃逸到 stderr。
    直接 patch handle_exception 本身，在任何 handler 之前就拦截，彻底消除打印。
    同时安装 asyncio 事件循环异常 handler 作为安全网。
    """
    _original_handle = app.handle_exception

    def _filtered_handle(exception: Exception) -> None:
        if isinstance(exception, RuntimeError) and "parent slot" in str(exception):
            return  # 静默吞掉，不传递给任何 handler
        _original_handle(exception)

    app.handle_exception = _filtered_handle

    def _on_startup():
        loop = asyncio.get_running_loop()
        _default_handler = loop.get_exception_handler()

        def _loop_exception_handler(loop, ctx):
            exc = ctx.get("exception")
            if isinstance(exc, RuntimeError) and "parent slot" in str(exc):
                return
            if _default_handler:
                _default_handler(loop, ctx)
            else:
                loop.default_exception_handler(ctx)

        loop.set_exception_handler(_loop_exception_handler)

    app.on_startup(_on_startup)


def main():
    """主函数"""
    _install_exception_filter()

    # 设置页面路由
    ui.page("/")(home_page)
    ui.page("/import")(import_page)
    ui.page("/split")(split_page)
    ui.page("/tagger")(tagger_page)
    ui.page("/caption")(caption_page)
    ui.page("/export")(export_page)
    ui.page("/tools")(tools_page)
    ui.page("/setup")(setup_page)
    ui.page("/console")(console_page)

    # 设置应用标题
    app.config.title = t("app_title")

    # 404 页面 - 捕获所有未定义的路由
    ui.page("/{path:path}")(not_found_page)

    # 运行应用
    ui.run(
        title=t("app_title"),
        favicon="🐉",
        dark=False,
        reload=False,
        port=8080,
        show=True,
        native=False,
        reconnect_timeout=30.0,
    )


if __name__ == "__main__":
    main()
