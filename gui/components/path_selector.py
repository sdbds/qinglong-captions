"""路径选择组件 - 现代化样式"""
from nicegui import ui
from pathlib import Path
from typing import Optional, Callable
from theme import get_classes, COLORS
from gui.utils.i18n import t


class PathSelector:
    """路径选择组件，支持文件和文件夹 - 现代化样式"""
    
    def __init__(self, 
                 label: str = "路径",
                 default_path: str = "",
                 selection_type: str = "file",
                 file_filter: Optional[str] = None,
                 on_change: Optional[Callable[[str], None]] = None,
                 placeholder: str = "点击选择路径..."):
        self.label = label
        self.selection_type = selection_type
        self.file_filter = file_filter
        self.on_change = on_change
        
        with ui.column().classes('w-full gap-2'):
            # 标签
            with ui.row().classes('items-center gap-2'):
                ui.icon('folder_open', size='18px').style(f'color: {COLORS["primary"]};')
                ui.label(label).classes('text-caption text-weight-medium').style('color: var(--color-text);')
            
            with ui.row().classes('w-full items-center gap-2'):
                # 输入框 - 现代化面包屑样式
                self.input = ui.input(value=default_path, placeholder=t('path_placeholder'))
                self.input.classes('flex-grow modern-input path-input')
                self.input.props('dense')
                if on_change:
                    self.input.on('change', lambda e: on_change(e.value))
                
                # 浏览按钮
                browse_btn = ui.button(icon='folder_open', on_click=self._pick_path)
                browse_btn.classes('modern-btn-secondary')
                browse_btn.props('dense').tooltip(t('browse'))
                
                # 更多操作菜单
                menu_btn = ui.button(icon='more_vert')
                menu_btn.classes('modern-btn-ghost')
                menu_btn.props('dense')
                
                with menu_btn:
                    with ui.menu().classes('q-pa-sm'):
                        ui.menu_item('📋 ' + t('copy'), self._copy_path)
                        ui.menu_item('📂 ' + t('open_folder', 'Open Folder'), self._open_folder)
                        ui.separator()
                        ui.menu_item('🗑️ ' + t('clear'), self._clear)
    
    async def _pick_path(self):
        """打开文件选择对话框"""
        if self.selection_type == "file":
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            if self.file_filter:
                # 转换文件过滤器格式
                patterns = self.file_filter.replace('*', '').replace(' ', ';').split(';')
                filetypes = [(f"{p} files", f"*{p}") for p in patterns if p] + [("All files", "*.*")]
            else:
                filetypes = [("All files", "*.*")]
            
            path = filedialog.askopenfilename(filetypes=filetypes)
            root.destroy()
            
        elif self.selection_type == "dir":
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askdirectory()
            root.destroy()
            
        elif self.selection_type == "save":
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.asksaveasfilename()
            root.destroy()
        else:
            path = ""
        
        if path:
            self.input.value = path
            self.input.update()
            if self.on_change:
                self.on_change(path)
    
    def _copy_path(self):
        """复制路径到剪贴板"""
        ui.run_javascript(f'navigator.clipboard.writeText(`{self.input.value}`)')
        ui.notify('✅ 路径已复制', type='positive')
    
    def _open_folder(self):
        """在文件管理器中打开文件夹"""
        import platform
        import subprocess
        
        path = Path(self.input.value)
        if path.is_file():
            path = path.parent
        
        if not path.exists():
            ui.notify('⚠️ 路径不存在', type='warning')
            return
        
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.run(["explorer", str(path)])
            elif system == "Darwin":
                subprocess.run(["open", str(path)])
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception as e:
            ui.notify(f'❌ 打开失败: {e}', type='negative')
    
    def _clear(self):
        """清空路径"""
        self.input.value = ""
        self.input.update()
        if self.on_change:
            self.on_change("")
    
    @property
    def value(self) -> str:
        """获取当前路径值"""
        return self.input.value
    
    @value.setter
    def value(self, val: str):
        """设置路径值"""
        self.input.value = val
        self.input.update()


def create_path_selector(**kwargs) -> PathSelector:
    """创建路径选择器的便捷函数"""
    return PathSelector(**kwargs)
