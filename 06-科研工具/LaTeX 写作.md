# LaTeX 写作

> 学术论文排版的事实标准——机器人/ML 方向必备

## 环境配置

### Overleaf (推荐新手)
- 在线编辑，多人协作
- 模板丰富（ICRA/IROS/NeurIPS 等）
- 缺点：大量编译慢

### 本地安装
```bash
# Windows: MiKTeX
# Linux: sudo apt install texlive-full
# Mac: brew install mactex
```

## 机器人/ML 常用模板

| 会议 | 模板 |
|------|------|
| ICRA / IROS | IEEEtran |
| CoRL | PMLR |
| NeurIPS | neurips_2024.sty |
| ICML | icml2024.sty |
| RSS | RSS 官方模板 |

## 常用技巧

### 公式
```latex
$$  % display math
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(o,a)\sim\mathcal{D}}
\left[\|\pi_\theta(o) - a\|^2\right]
$$

$\theta$  % inline math
```

### 算法伪代码
```latex
\usepackage{algorithm}
\usepackage{algpseudocode}

\begin{algorithm}
\caption{Behavior Cloning Training}
\begin{algorithmic}
\For{each epoch}
    \For{each batch $(o, a) \sim \mathcal{D}$}
        \State $\hat{a} \leftarrow \pi_\theta(o)$
        \State $\mathcal{L} \leftarrow \text{MSE}(\hat{a}, a)$
        \State $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$
    \EndFor
\EndFor
\end{algorithmic}
\end{algorithm}
```

### 引用
```latex
\cite{brohan2022rt1}      → [1]
\citep{brohan2022rt1}     → (Brohan et al., 2022)
```

### 图表
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\linewidth]{figs/architecture.pdf}
    \caption{VLA 模型架构图}
    \label{fig:arch}
\end{figure}
```

## 常见问题

| 问题 | 解决 |
|------|------|
| 图片不显示 | 检查路径 + 编译两次 |
| 引用是 ?? | 多编译一次 (pdflatex → bibtex → pdflatex → pdflatex) |
| 表格太宽 | 用 `\resizebox{\linewidth}{!}{...}` |
| 中文支持 | 用 `\usepackage[UTF8]{ctex}` |

## 工具链

- [Tables Generator](https://www.tablesgenerator.com/) — 可视化画 LaTeX 表格
- [Mathpix](https://mathpix.com/) — 截图转 LaTeX 公式
- [Detexify](http://detexify.kirelabs.org/) — 手写识别符号

## 关联笔记
- [[AI辅助论文写作]] — AI 帮写帮改
- [[论文绘图工具]] — 图表准备
- [[参考文献管理]] — .bib 文件管理
