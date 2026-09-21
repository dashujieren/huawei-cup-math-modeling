# 华为杯研究生数学建模团队仓库设计

## 目标

建立一个长期保存在 `D:\codex\huawei-cup-math-modeling` 的 Git 仓库，并将其发布为 GitHub 私有仓库 `huawei-cup-math-modeling`。仓库用于团队集中保存比赛题面、数据、代码、实验记录、图表、论文与协作记录，同时保证关键结果可复现、原始材料不被覆盖、敏感信息不进入版本历史。

## 已确认边界

- 远程平台为 GitHub，仓库可见性为 private。
- 本地默认分支为 `main`。
- GitHub 仓库归属当前完成网页授权的账号。
- 首次交付只创建仓库、基础目录、协作说明和远程连接，不替团队选择赛题、模型或论文结论。
- 当前没有官方题面和当届提交规则，因此不预设页数、格式、模型数量或交付文件名。
- 队友账号尚未提供，首次交付不自动邀请协作者；仓库建立后可按 GitHub 用户名逐一添加。

## 创建方式

使用 GitHub 官方 CLI 的 Windows 便携版完成浏览器授权、私有仓库创建和首次推送。CLI 放在 `D:\codex\tools\github-cli`，不进入比赛仓库。相比手工创建空仓库，此方式能把远程创建、`origin` 配置和首次推送放在同一条可验证流程中，也不要求用户在对话中传递访问令牌。

## 仓库结构

```text
huawei-cup-math-modeling/
├─ README.md
├─ CONTRIBUTING.md
├─ .gitignore
├─ .gitattributes
├─ requirements.txt
├─ PROJECT_BRIEF.md
├─ problem/
│  └─ official/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  └─ README.md
├─ src/
│  ├─ preprocessing/
│  ├─ models/
│  ├─ evaluation/
│  └─ visualization/
├─ scripts/
├─ configs/
├─ notebooks/
├─ tests/
├─ outputs/
│  ├─ runs/
│  └─ tables/
├─ figures/
├─ paper/
│  └─ assets/
├─ references/
├─ notes/
│  ├─ meetings/
│  └─ decisions.md
└─ docs/
   └─ superpowers/specs/
```

空目录通过用途说明文件保留。正式计算代码应位于 `src/`，`notebooks/` 只用于探索，不作为最终结果的唯一来源。`problem/official/` 与 `data/raw/` 中的原始文件按只读原则使用，任何清洗或转换结果写入后续目录。

## Git 与协作约定

- `main` 只接收可运行、已检查的内容；每位队员使用短期功能分支，并通过合并请求或审阅后合并。
- 提交信息采用简短的动作说明，例如 `model: add baseline regression`、`paper: revise assumptions`。
- 每次正式实验记录参数、随机种子、依赖版本和对应 Git 提交号；历史实验结果不得静默覆盖。
- 不提交访问令牌、账号密码、队伍身份信息、缓存、编辑器状态、LaTeX 编译垃圾或可重新生成的超大中间文件。

## 大文件策略

环境已安装 Git LFS。初始 `.gitattributes` 对常见大型二进制数据、模型文件和办公文档启用 LFS；文本、源代码、LaTeX 与小型配置继续由普通 Git 管理。对于超过 GitHub/LFS 配额或禁止公开分发的数据，只提交来源说明、获取方法与校验值，不提交文件本体。

## 初始化内容

- `README.md`：仓库入口、快速开始、目录导航和当前状态。
- `CONTRIBUTING.md`：分支、提交、评审、文件命名和冲突处理约定。
- `.gitignore`：覆盖 Windows、Python、MATLAB、Jupyter、LaTeX、IDE、密钥和临时输出。
- `.gitattributes`：统一文本换行并声明 Git LFS 类型。
- `requirements.txt`：保持为空依赖集合的有效说明；在代码引入第三方库时固定版本。
- `PROJECT_BRIEF.md`：只记录已知比赛背景与未确认的官方要求，不把猜测写成事实。
- 各数据、代码、结果和论文目录：提供最小用途说明，不放示例假数据或虚构结论。

## 验证标准

完成后必须满足：

1. `git status` 干净，默认分支为 `main`，至少有一个包含仓库骨架的提交。
2. Git LFS 初始化成功，声明的文件模式可由 `git lfs track` 识别。
3. GitHub 上存在同名 private 仓库，`origin` 指向该仓库。
4. 本地 `main` 已推送并跟踪 `origin/main`。
5. GitHub CLI 查询确认仓库可见性为 private。
6. 仓库中不存在访问令牌、密码或真实队伍身份信息。

## 后续使用

收到当届官方题面、附件和提交规则后，先把用户提供的原始文件做只读清点，再补全 `PROJECT_BRIEF.md`，并按实际题目进入分析、建模、计算、验证和论文阶段。仓库骨架本身不代表已经核验任何当届官方规则。
