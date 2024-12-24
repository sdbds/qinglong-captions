#!/usr/bin/bash
#export PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
#export HF_ENDPOINT="https://hf-mirror.com"

echo "检查是否已安装 PowerShell..."
if ! command -v pwsh &> /dev/null
then
    echo "PowerShell 未安装，正在安装..."
    
    # 下载 PowerShell '.tar.gz' 压缩包
    curl -L -o /tmp/powershell.tar.gz https://mirror.ghproxy.com/https://github.com/PowerShell/PowerShell/releases/download/v7.4.6/powershell-7.4.6-linux-x64.tar.gz
    
    # 创建目标文件夹
    sudo mkdir -p /opt/microsoft/powershell/7
    
    # 解压 PowerShell 到目标文件夹
    sudo tar zxf /tmp/powershell.tar.gz -C /opt/microsoft/powershell/7
    
    # 设置执行权限
    sudo chmod +x /opt/microsoft/powershell/7/pwsh
    
    # 创建指向 pwsh 的符号链接
    sudo ln -s /opt/microsoft/powershell/7/pwsh /usr/bin/pwsh
    
    echo "PowerShell 安装完成"
else
    echo "PowerShell 已安装"
fi

echo "Install completed"
