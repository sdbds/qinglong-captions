#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
控制台工具类
提供Rich布局和显示功能
"""

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.markdown import Markdown

# 全局控制台实例
console = Console()


class BaseLayout:
    """基础布局类，提供创建Rich布局的基本功能"""
    
    def __init__(self, panel_height=32, console=None):
        """
        初始化基础布局
        
        Args:
            panel_height: 面板高度
            console: Rich控制台实例
        """
        self.panel_height = panel_height
        self.console = console or globals().get('console', Console())
        self.layout = Layout()
    
    def create_layout(self):
        """创建基本布局结构（由子类实现）"""
        pass
    
    def render(self, title=""):
        """
        渲染布局为面板并返回
        
        Args:
            title: 面板标题
            
        Returns:
            Panel: Rich面板对象
        """
        panel = Panel(
            self.layout,
            title=title,
            height=self.panel_height + 2,
            padding=0,
        )
        return panel
    
    def print(self, title=""):
        """
        打印布局到控制台
        
        Args:
            title: 面板标题
        """
        panel = self.render(title)
        self.console.print()
        self.console.print()
        self.console.print(panel)


class CaptionLayout(BaseLayout):
    """用于显示图片字幕的布局类"""
    
    def __init__(self, tag_description, short_description, long_description, pixels,
                short_highlight_rate=0, long_highlight_rate=0, panel_height=32, console=None):
        """
        初始化字幕布局
        
        Args:
            tag_description: 标签描述
            short_description: 短描述
            long_description: 长描述
            pixels: Rich Pixels对象
            short_highlight_rate: 短描述高亮率
            long_highlight_rate: 长描述高亮率
            panel_height: 面板高度
            console: Rich控制台实例
        """
        super().__init__(panel_height, console)
        self.tag_description = tag_description
        self.short_description = short_description
        self.long_description = long_description
        self.pixels = pixels
        self.short_highlight_rate = short_highlight_rate
        self.long_highlight_rate = long_highlight_rate
        self.create_layout()
    
    def create_layout(self):
        """创建字幕布局结构"""
        # 创建右侧的垂直布局
        right_layout = Layout()
        
        # 创建上半部分的水平布局（tag和short并排）
        top_layout = Layout()
        top_layout.split_row(
            Layout(
                Panel(
                    Text(self.tag_description, style="magenta"),
                    title="tags",
                    height=self.panel_height // 2,
                    padding=0,
                    expand=True,
                ),
                ratio=1,
            ),
            Layout(
                Panel(
                    self.short_description,
                    title=f"short_description - [yellow]highlight rate:[/yellow] {self.short_highlight_rate}",
                    height=self.panel_height // 2,
                    padding=0,
                    expand=True,
                ),
                ratio=1,
            ),
        )
        
        # 将右侧布局分为上下两部分
        right_layout.split_column(
            Layout(top_layout, ratio=1),
            Layout(
                Panel(
                    self.long_description,
                    title=f"long_description - [yellow]highlight rate:[/yellow] {self.long_highlight_rate}",
                    height=self.panel_height // 2,
                    padding=0,
                    expand=True,
                )
            ),
        )
        
        # 主布局分为左右两部分
        self.layout.split_row(
            Layout(
                Panel(self.pixels, height=self.panel_height, padding=0, expand=True),
                name="image",
                ratio=1,
            ),
            Layout(right_layout, name="caption", ratio=2),
        )


class MarkdownLayout(BaseLayout):
    """用于显示Markdown内容的布局类"""
    
    def __init__(self, pixels, markdown_content, panel_height=32, console=None):
        """
        初始化Markdown布局
        
        Args:
            pixels: Rich Pixels对象
            markdown_content: Markdown内容
            panel_height: 面板高度
            console: Rich控制台实例
        """
        super().__init__(panel_height, console)
        self.pixels = pixels
        self.markdown_content = markdown_content
        self.create_layout()
    
    def create_layout(self):
        """创建Markdown布局结构"""
        # 创建右侧布局（单个Markdown窗口）
        right_layout = Layout(
            Panel(
                Markdown(self.markdown_content),
                title="markdown",
                padding=0,
                expand=True,
            )
        )
        
        # 如果pixels为空，直接全局渲染markdown内容，否则分为左右两部分
        if self.pixels is None:
            self.layout.update(
                Layout(
                    Panel(
                        Markdown(self.markdown_content),
                        title="markdown",
                        padding=0,
                        expand=True,
                    ),
                    name="markdown",
                )
            )
        else:
            self.layout.split_row(
                Layout(
                    Panel(self.pixels, height=self.panel_height, padding=0, expand=True),
                    name="image",
                    ratio=1,
                ),
                Layout(right_layout, name="markdown", ratio=2),
            )
