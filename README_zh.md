# 构建 nanoGPT

这个仓库包含了从头开始复现 [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) 的过程。Git 提交记录被特意保持得清晰且循序渐进，这样你可以通过浏览 git 提交历史来了解它是如何一步步构建的。此外，还有配套的 [YouTube 视频讲座](https://youtu.be/l8pRSuU81PU)，你可以在视频中看到我介绍每个提交并解释其中的细节。

我们基本上是从一个空文件开始，最终复现了 [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (124M) 模型。如果你有更多的耐心或资金，这段代码也可以用来复现 [GPT-3](https://arxiv.org/pdf/2005.14165) 模型。虽然 GPT-2 (124M) 模型在当时（2019年，约5年前）训练可能需要相当长的时间，但今天复现它只需要约1小时和约10美元。如果你没有足够的计算资源，你需要一个云 GPU 服务器，为此我推荐使用 [Lambda](https://lambdalabs.com)。

请注意，GPT-2 和 GPT-3 都是简单的语言模型，在互联网文档上训练，它们所做的就是"生成"互联网文档。所以这个仓库/视频不涉及聊天微调，你不能像和 ChatGPT 那样与它对话。微调过程（虽然在概念上很简单 - SFT 基本上就是更换数据集并继续训练）是在这部分之后的内容，将在之后介绍。目前，如果你用"Hello, I'm a language model"来提示这个 124M 的模型，在经过 10B tokens 的训练后，它会这样回答：

```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
Hello, I'm a language model, and I'm an English person. In languages, "speak" is really speaking. Because for most people, there's
```

在经过 40B tokens 的训练后：

```
Hello, I'm a language model, a model of computer science, and it's a way (in mathematics) to program computer programs to do things like write
Hello, I'm a language model, not a human. This means that I believe in my language model, as I have no experience with it yet.
Hello, I'm a language model, but I'm talking about data. You've got to create an array of data: you've got to create that.
Hello, I'm a language model, and all of this is about modeling and learning Python. I'm very good in syntax, however I struggle with Python due
```

哈哈。总之，视频发布后，这里也将成为 FAQ 和修复与勘误的地方，我相信会有很多需要补充的内容 :)

如果你有任何讨论和问题，请使用 [Discussions 标签](https://github.com/karpathy/build-nanogpt/discussions)，如果想要更快的交流，可以加入我的 [Zero To Hero Discord](https://discord.gg/3zy8kqD9Cp)，在 **#nanoGPT** 频道：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 视频

[让我们复现 GPT-2 (124M) YouTube 讲座](https://youtu.be/l8pRSuU81PU)

## 勘误

一些小清理，我们在切换到 flash attention 后忘记删除偏置的 `register_buffer`，这个问题已经在最近的 PR 中修复。

较早版本的 PyTorch 可能在从 uint16 转换为 long 时遇到困难。在 `load_tokens` 中，我们添加了 `npt = npt.astype(np.int32)` 来使用 numpy 先将 uint16 转换为 int32，然后再转换为 torch tensor 并最终转换为 long。

`torch.autocast` 函数接受一个 `device_type` 参数，我曾固执地尝试直接传递 `device` 希望它能正常工作，但 PyTorch 实际上只想要类型，在某些版本的 PyTorch 中会创建错误。所以我们需要例如将设备 `cuda:3` 转换为 `cuda`。目前，设备 `mps`（Apple Silicon）会变成 `device_type` CPU，我不确定这是否是 PyTorch 的预期方式。

令人困惑的是，`model.require_backward_grad_sync` 实际上在前向和后向传播中都被使用。我们将这行代码移到了前面，这样它也能应用到前向传播中。

## 生产环境

如果你想要更接近生产级别的运行环境，与 nanoGPT 非常相似，我推荐查看以下仓库：

- [litGPT](https://github.com/Lightning-AI/litgpt)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)

## 常见问题

## 许可证

MIT
