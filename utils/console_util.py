from rich.console import Console
from rich.layout import Layout
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from rich_pixels import Pixels

# 全局控制台实例
console = Console(color_system="truecolor", force_terminal=True)


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
        self.console = console or globals().get("console", Console())
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

    def __init__(
        self,
        tag_description,
        short_description,
        long_description,
        pixels,
        short_highlight_rate=0,
        long_highlight_rate=0,
        panel_height=32,
        console=None,
    ):
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
        from utils.wdtagger import TagClassifier

        tagClassifier = TagClassifier()
        # Process tag_description to handle various spacing and comma combinations
        cleaned_description = tag_description.replace("<", "").replace(">", "")
        processed_tags = [tag.strip() for tag in cleaned_description.split(",") if tag.strip()]
        tag_values = tagClassifier.classify(processed_tags).values()
        self.tag_description = ",".join([",".join(value) for value in tag_values])
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
                    self.tag_description,
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
                    title=f"short_description - [yellow]hr:[/yellow] {self.short_highlight_rate}",
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


class CaptionAndRateLayout(BaseLayout):
    """用于显示图片字幕和评分的布局类"""

    def __init__(
        self,
        tag_description,
        rating,
        average_score,
        long_description,
        pixels,
        short_highlight_rate=0,
        long_highlight_rate=0,
        panel_height=32,
        console=None,
    ):
        """
        初始化字幕布局

        Args:
            tag_description: 标签描述
            rating: 高亮率
            average_score: 平均评分
            long_description: 长描述
            pixels: Rich Pixels对象
            long_highlight_rate: 长描述高亮率
            panel_height: 面板高度
            console: Rich控制台实例
        """
        super().__init__(panel_height, console)
        self.tag_description = tag_description
        self.long_description = long_description
        self.long_highlight_rate = long_highlight_rate
        self.pixels = pixels
        self.rating_chart = self.create_rating_chart(rating)
        self.average_score = average_score
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
                    self.rating_chart,
                    title=f"rating - [yellow]average score:[/yellow] {self.average_score}",
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

    def create_rating_chart(self, ratings, max_rating=10):
        """创建一个简单的评分图

        Args:
            ratings: 字典，包含维度名称和对应评分
            max_rating: 最大评分值
        """
        if not ratings:
            return Pixels.from_ascii("No ratings available")

        # 获取维度列表并清理维度名称
        clean_ratings = {}
        for key, value in ratings.items():
            # 移除所有非标准字符，不仅仅是方块字符
            clean_key = ""
            for char in key:
                # 只保留字母、数字、空格和常见标点符号
                if char.isalnum() or char.isspace() or char in "&/,.:-_()":
                    clean_key += char
            # 如果清理后为空，使用原始键
            if not clean_key:
                clean_key = f"Dimension {len(clean_ratings) + 1}"
            clean_ratings[clean_key] = value

        # 创建彩虹颜色映射
        rainbow_colors = [
            "bright_red",  # 红色
            "orange3",  # 橙色
            "yellow",  # 黄色
            "green",  # 绿色
            "spring_green3",  # 青色
            "bright_blue",  # 蓝色
            "blue_violet",  # 靛色
            "purple",  # 紫色
            "magenta",  # 紫红色
        ]

        # 准备行和颜色映射
        lines = []
        mapping = {}

        # 为每个维度创建一行
        for i, (dimension, rating) in enumerate(clean_ratings.items()):
            # 获取对应颜色
            color_index = min(i, len(rainbow_colors) - 1)
            color = rainbow_colors[color_index]

            # 处理维度名称，只保留第一个&前的内容
            short_dim = dimension.split("&")[0].strip()
            dim_text = f"{short_dim}:"
            # 评分条长度等于评分值
            bar_length = int(rating)

            # 使用隐藏的控制字符作为映射键，这些字符在普通文本中不太可能出现
            # 使用ASCII 01-31的不可见控制字符，每行使用不同的控制字符
            control_char = chr(1 + i)  # 使用 SOH, STX, ETX 等不可见控制字符
            mapping[control_char] = Segment("■", Style(color=color))

            # 生成评分条
            bar = control_char * bar_length

            # 特殊处理某些维度的最大分数
            current_max_rating = 5 if dimension in ["Storytelling & Concept", "Setting & Environment Integration"] else max_rating
            value_text = f" {rating}/{current_max_rating}"

            # 组合行内容
            line = dim_text + bar + value_text
            lines.append(line)

        # 组合成ASCII图
        ascii_grid = "\n".join(lines)

        # 返回Pixels对象
        return Pixels.from_ascii(ascii_grid, mapping)


class CaptionPairImageLayout(BaseLayout):
    """A layout for displaying two images side-by-side with a description in the middle."""

    def __init__(
        self,
        description: str,
        pixels: "Pixels",
        pair_pixels: "Pixels",
        panel_height: int = 32,
        console: "Console" = None,
    ):
        """
        Initializes the layout for displaying a pair of captions.

        Args:
            description: The description text.
            pixels: The Rich Pixels object for the first image.
            pair_pixels: The Rich Pixels object for the second image.
            panel_height: The height of the panel.
            console: The Rich console instance.
        """
        super().__init__(panel_height, console)
        self.description = description
        self.pixels = pixels
        self.pair_pixels = pair_pixels
        self.create_layout()

    def create_layout(self):
        """Creates the layout structure with three columns."""
        # Main layout splits into three columns
        self.layout.split_row(
            Layout(
                Panel(
                    self.pair_pixels,
                    title="Original",
                    height=self.panel_height,
                    padding=0,
                    expand=True,
                ),
                name="image1",
                ratio=2,
            ),
            Layout(
                Panel(
                    Text(self.description, justify="center"),
                    title="Description",
                    height=self.panel_height,
                    padding=1,
                    expand=True,
                ),
                name="description",
                ratio=1,
            ),
            Layout(
                Panel(
                    self.pixels,
                    title="Edited",
                    height=self.panel_height,
                    padding=0,
                    expand=True,
                ),
                name="image2",
                ratio=2,
            ),
        )
