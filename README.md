# 🤖 UltraGPT

<div align="center">

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)](https://openai.com/)

**A powerful and modular library for advanced GPT-based reasoning and step pipelines**

</div>

## 🌟 Features

- **📝 Steps Pipeline:** Break down complex tasks into manageable steps
  - Automatic step generation and processing
  - Verification at each step
  - Detailed progress tracking

- **🧠 Reasoning Pipeline:** Advanced reasoning capabilities
  - Multi-iteration thought process
  - Building upon previous reasoning
  - Comprehensive analysis

- **🛠️ Tool Integration:** 
  - Web search capabilities
  - Calculator functionality
  - Extensible tool framework

## 📦 Installation

```bash
pip install git+https://github.com/Kawai-Senpai/UltraGPT.git
```

## 🚀 Quick Start

```python
from ultragpt import UltraGPT

if __name__ == "__main__":
    # Initialize UltraGPT
    ultragpt = UltraGPT(
        api_key="your-openai-api-key",
        verbose=True
    )

    # Example chat session
    final_output, tokens_used, details = ultragpt.chat([
        {"role": "user", "content": "Write a story about an elephant."}
    ])

    print("Final Output:", final_output)
    print("Total tokens used:", tokens_used)
```

## 📚 Advanced Usage

### Customizing Pipeline Settings

```python
ultragpt = UltraGPT(
    api_key="your-openai-api-key",
    model="gpt-4o",  # Specify model
    temperature=0.7,  # Adjust creativity
    reasoning_iterations=3,  # Set reasoning depth
    steps_pipeline=True,
    reasoning_pipeline=True,
    verbose=True
)
```

### Using Tools

```python
ultragpt = UltraGPT(
    api_key="your-openai-api-key",
    tools=["web-search", "calculator"],
    tools_config={
        "web-search": {
            "max_results": 1,
            "model": "gpt-4o"
        },
        "calculator": {
            "model": "gpt-4o"
        }
    }
)
```

## 🔧 Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | Required | Your OpenAI API key |
| `model` | str | "gpt-4o" | Model to use |
| `temperature` | float | 0.7 | Output randomness |
| `reasoning_iterations` | int | 3 | Number of reasoning steps |
| `tools` | list | [] | Enabled tools |
| `verbose` | bool | False | Enable detailed logging |

## 🌐 Tool System

UltraGPT supports various tools to enhance its capabilities:

### Web Search
- Performs intelligent web searches
- Summarizes findings
- Integrates results into responses

### Calculator
- Handles mathematical operations
- Supports complex calculations
- Provides step-by-step solutions

## 🔄 Pipeline System

### Steps Pipeline
1. Task Analysis
2. Step Generation
3. Step-by-Step Execution
4. Progress Verification
5. Final Compilation

### Reasoning Pipeline
1. Initial Analysis
2. Multi-iteration Thinking
3. Thought Development
4. Conclusion Formation

## 📋 Requirements

- Python 3.6+
- OpenAI API key
- Internet connection (for web tools)

## 🤝 Contributing

Contributions are always welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes
4. Commit (`git commit -am 'Add new feature'`)
5. Push (`git push origin feature/improvement`)
6. Open a Pull Request

## 📝 License

This project is MIT licensed - see the [LICENSE](LICENSE) file for details.

## 👥 Author

**Ranit Bhowmick**
- Email: bhowmickranitking@duck.com
- GitHub: [@Kawai-Senpai](https://github.com/Kawai-Senpai)

---

<div align="center">
Made with ❤️ by Ranit Bhowmick
</div>
