"""步骤 3: 数据集打标 - 对应 tagger.ps1"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.log_viewer import create_log_viewer
from components.advanced_inputs import editable_slider, toggle_switch, styled_select, styled_input
from gui.utils.job_manager import job_manager
from gui.utils.process_runner import ProcessStatus
from gui.utils.i18n import t


class TaggerStep:
    """数据集打标页面"""

    DEFAULT_MODELS = [
        "cella110n/cl_tagger",
        "SmilingWolf/wd-eva02-large-tagger-v3",
        "SmilingWolf/wd-vit-large-tagger-v3",
        "SmilingWolf/wd-vit-tagger-v3",
        "SmilingWolf/wd-swinv2-tagger-v3",
    ]

    def __init__(self):
        self.config: Dict[str, Any] = {
            "batch_size": 12,
            "thresh": 0.6,
            "general_threshold": 0.55,
            "character_threshold": 1.0,
            "remove_underscore": True,
            "frequency_tags": False,
            "use_rating_tags": True,
            "use_quality_tags": False,
            "use_model_tags": False,
            "character_tags_first": False,
            "remove_parents_tag": True,
            "overwrite": True,
        }
        self.log_viewer = None
        self.is_running = False
        self.current_job = None

    def render(self):
        """渲染页面"""
        with ui.column().classes(get_classes("page_container") + " gap-4"):
            # 页面标题
            with ui.row().classes("w-full items-center gap-3 q-mb-sm"):
                ui.icon("label", size="32px").style(f"color: {COLORS['primary']};")
                with ui.column().classes("gap-0"):
                    ui.label(t("tagger_title")).classes("text-h4 text-weight-bold").style("color: var(--color-text);")
                    ui.label(t("tagger_desc")).classes("text-body2").style("color: var(--color-text-secondary);")

            with ui.stepper().props("vertical").classes("w-full") as stepper:
                # 步骤 3.1: 配置路径和模型
                with ui.step(t("config_paths"), icon="folder_open"):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("folder_open", size="22px").style(f"color: {COLORS['info']};")
                            ui.label(t("dataset_path")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                        # 训练数据目录
                        self.train_data_dir = create_path_selector(
                            label=t("train_data_dir"), selection_type="dir", placeholder=t("input_path_placeholder")
                        )

                        # 模型仓库 - 带图标的现代化下拉框
                        self.repo_id = styled_select(
                            options=dict(zip(self.DEFAULT_MODELS, self.DEFAULT_MODELS)),
                            value="cella110n/cl_tagger",
                            label=t("model_repo"),
                            icon="model_training",
                            icon_color=COLORS["primary"],
                            new_value_mode="add-unique",
                        )

                        # 模型目录
                        self.model_dir = styled_input(
                            value="wd14_tagger_model", label=t("model_dir"), icon="folder", icon_color=COLORS["primary"]
                        )

                    with ui.row().classes("w-full justify-end q-mt-md"):
                        next_btn = ui.button(t("next_step"), on_click=stepper.next, icon="arrow_forward")
                        next_btn.classes("modern-btn-primary").props('type="button"')

                # 步骤 3.2: 配置参数
                with ui.step(t("tagging_settings"), icon="tune"):
                    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                        with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                            ui.icon("tune", size="22px").style(f"color: {COLORS['warning']};")
                            ui.label(t("tagging_settings")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

                        # 使用可编辑滑块替代数字输入
                        with ui.row().classes("w-full gap-4"):
                            editable_slider(
                                label_key="batch_size",
                                value_ref=self.config,
                                value_key="batch_size",
                                min_val=1,
                                max_val=64,
                                step=1,
                                decimals=0,
                            )

                            editable_slider(
                                label_key="thresh",
                                value_ref=self.config,
                                value_key="thresh",
                                min_val=0.0,
                                max_val=1.0,
                                step=0.05,
                                decimals=2,
                            )

                        with ui.row().classes("w-full gap-4 q-mt-md"):
                            editable_slider(
                                label_key="general_threshold",
                                value_ref=self.config,
                                value_key="general_threshold",
                                min_val=0.0,
                                max_val=1.0,
                                step=0.05,
                                decimals=2,
                            )

                            editable_slider(
                                label_key="character_threshold",
                                value_ref=self.config,
                                value_key="character_threshold",
                                min_val=0.0,
                                max_val=1.0,
                                step=0.05,
                                decimals=2,
                            )

                        # 功能开关 - 使用按钮式开关
                        with ui.card().classes(get_classes("card") + " w-full q-pa-md q-mt-md"):
                            ui.label(t("feature_toggles")).classes("text-subtitle1 text-weight-bold").style("color: var(--color-text);")

                            with ui.grid(columns=3).classes("w-full gap-4 q-mt-sm"):
                                toggle_switch("remove_underscore", self.config, "remove_underscore")
                                toggle_switch("frequency_tags", self.config, "frequency_tags")
                                toggle_switch("use_rating_tags", self.config, "use_rating_tags")
                                toggle_switch("use_quality_tags", self.config, "use_quality_tags")
                                toggle_switch("use_model_tags", self.config, "use_model_tags")
                                toggle_switch("character_tags_first", self.config, "character_tags_first")
                                toggle_switch("remove_parents_tag", self.config, "remove_parents_tag")
                                toggle_switch("overwrite", self.config, "overwrite")

                        # 高级标签设置
                        with ui.expansion(t("advanced_tag_settings")).classes("w-full q-mt-md"):
                            self.undesired_tags = ui.input(label=t("undesired_tags"), placeholder="tag1,tag2,tag3")
                            self.undesired_tags.classes("modern-input w-full")

                            self.always_first_tags = ui.input(
                                label=t("always_first_tags"),
                                value="1girl,1boy,2girls,3girls,4girls,5girls,6girls,2boys,3boys,4boys,5boys,6boys",
                            )
                            self.always_first_tags.classes("modern-input w-full")

                            self.tag_replacement = ui.input(
                                label=t("tag_replacement"),
                                value="1girl,1woman;2girls,2women;3girls,3women;4girls,4women;5girls,5women;1boy,1man",
                            )
                            self.tag_replacement.classes("modern-input w-full")

                    with ui.row().classes("w-full items-center justify-between q-mt-md"):
                        prev_btn = ui.button(t("prev_step"), on_click=stepper.previous, icon="arrow_back")
                        prev_btn.classes("modern-btn-ghost").props('type="button"')

                        with ui.row().classes("gap-2"):
                            self.stop_btn = ui.button(t("stop"), on_click=self._stop_tagging, icon="stop")
                            self.stop_btn.classes("modern-btn-danger").props('type="button"')
                            self.stop_btn.set_enabled(False)

                            self.start_btn = ui.button(t("start_tagging"), on_click=self._start_tagging, icon="play_arrow")
                            self.start_btn.classes("modern-btn-success").props('type="button"')

                    # 日志查看器
                    self.log_viewer = create_log_viewer()

    async def _start_tagging(self):
        """开始打标"""
        try:
            train_data_dir = self.train_data_dir.value
            if not train_data_dir or not Path(train_data_dir).exists():
                ui.notify(t("select_valid_train_dir"), type="warning")
                return

            self.is_running = True
            self.start_btn.set_enabled(False)
            self.stop_btn.set_enabled(True)

            repo_id = self.repo_id.value
            model_dir = self.model_dir.value or "wd14_tagger_model"
            batch_size = int(self.config["batch_size"])
            thresh = self.config["thresh"]
            general_threshold = self.config["general_threshold"]
            character_threshold = self.config["character_threshold"]

            # 构建参数
            args = [train_data_dir]
            args.append(f"--repo_id={repo_id}")
            args.append(f"--model_dir={model_dir}")
            args.append(f"--batch_size={batch_size}")
            args.append(f"--thresh={thresh}")
            args.append(f"--general_threshold={general_threshold}")
            args.append(f"--character_threshold={character_threshold}")
            args.append("--caption_extension=.txt")

            # 功能开关
            if self.config["remove_underscore"]:
                args.append("--remove_underscore")
            if self.config["frequency_tags"]:
                args.append("--frequency_tags")
            if self.config["use_rating_tags"]:
                args.append("--use_rating_tags")
            if self.config["use_quality_tags"]:
                args.append("--use_quality_tags")
            if self.config["use_model_tags"]:
                args.append("--use_model_tags")
            if self.config["character_tags_first"]:
                args.append("--character_tags_first")
            if self.config["remove_parents_tag"]:
                args.append("--remove_parents_tag")
            if self.config["overwrite"]:
                args.append("--overwrite")

            # 标签设置
            if self.undesired_tags.value:
                args.append(f"--undesired_tags={self.undesired_tags.value}")
            if self.always_first_tags.value:
                args.append(f"--always_first_tags={self.always_first_tags.value}")
            if self.tag_replacement.value:
                args.append(f"--tag_replacement={self.tag_replacement.value}")

            # 提交 Job（Tagger 使用 wdtagger extra）
            job = await job_manager.submit("utils.wdtagger", args, name="Tagger (WD14)")
            self.current_job = job
            self.log_viewer.attach_job(job)

            self.log_viewer.info(t("log_start_tagging"))
            self.log_viewer.info(f"{t('log_data_dir')}: {train_data_dir}")
            self.log_viewer.info(f"{t('log_model')}: {repo_id}")
            self.log_viewer.info(f"{t('log_batch_size')}: {batch_size}")
            self.log_viewer.info(f"{t('log_params')}: {args}")

            result = await job.wait()

            if result.status == ProcessStatus.SUCCESS:
                self.log_viewer.success(t("tagging_success"))
                ui.notify(t("tagging_success"), type="positive")
            else:
                self.log_viewer.error(f"{t('tagging_failed')}: {result.message}")
                ui.notify(t("tagging_failed"), type="negative")

        except RuntimeError:
            pass  # 用户已离开页面，元素已销毁
        except Exception as e:
            try:
                self.log_viewer.error(f"{t('tagging_error')}: {e}")
                ui.notify(f"{t('tagging_error')}: {e}", type="negative")
            except RuntimeError:
                pass
        finally:
            try:
                self.is_running = False
                self.current_job = None
                self.start_btn.set_enabled(True)
                self.stop_btn.set_enabled(False)
            except RuntimeError:
                pass

    def _stop_tagging(self):
        """停止打标"""
        if self.current_job:
            job_manager.cancel(self.current_job.id)
            self.current_job = None
        self.is_running = False
        self.start_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)
        self.log_viewer.info(t("task_stopped"))
        ui.notify(t("task_stopped"), type="info")


def render_tagger_step():
    """渲染打标步骤"""
    step = TaggerStep()
    step.render()
