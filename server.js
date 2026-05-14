// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const RAW_BASE = process.env.NIM_API_BASE;
const NIM_API_BASE = (RAW_BASE && RAW_BASE.trim() !== "") 
  ? RAW_BASE.trim().replace(/\/+$/, "") 
  : 'https://integrate.api.nvidia.com/v1'; 

const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output inside <think> tags
const SHOW_REASONING = true;

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = true;

// Model mapping - FIXED: Corrected to authentic NVIDIA NIM catalog IDs
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'meta/llama-3.1-405b-instruct',
  'gpt-4-turbo': 'meta/llama-3.3-70b-instruct',
  'gpt-4o': 'deepseek-ai/deepseek-v3',
  'claude-3-opus': 'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',
  'gemini-pro': 'deepseek-ai/deepseek-v3'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    api_base_used: NIM_API_BASE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream, top_p } = req.body;

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      const modelLower = model ? model.toLowerCase() : '';
      if (modelLower.includes('gpt-4') || modelLower.includes('opus') || modelLower.includes('405b')) {
        nimModel = 'meta/llama-3.1-405b-instruct';
      } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b') || modelLower.includes('3.3')) {
        nimModel = 'meta/llama-3.3-70b-instruct';
      } else {
        nimModel = 'nvidia/llama-3.1-nemotron-ultra-253b-v1';
      }
    }

    const isThinkingModel = nimModel.toLowerCase().includes('thinking') || nimModel.toLowerCase().includes('deepseek-r1');
    const activeTemperature = isThinkingModel ? 1.0 : (temperature || 0.7);

    // FIXED: Build clean request payload stripping JanitorAI client parameter bloat to prevent 500 errors
    const nimRequest = {
      model: nimModel,
      messages: messages || [],
      temperature: activeTemperature,
      max_tokens: max_tokens || 4096,
      top_p: top_p || 1,
      stream: stream || false
    };

    // Include stream options if streaming to safely parse usage metrics
    if (stream) {
      nimRequest.stream_options = { include_usage: true };
    }

    // Append thinking template modifications safely
    if (ENABLE_THINKING_MODE && isThinkingModel) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;
      let thinkingTagClosed = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        let lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          const trimmedLine = line.trim();
          if (!trimmedLine) return;
          if (!trimmedLine.startsWith('data: ')) return;

          if (trimmedLine.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            return;
          }

          try {
            const data = JSON.parse(trimmedLine.slice(6));
            if (data.choices && data.choices[0] && data.choices[0].delta) {
              const reasoning = data.choices[0].delta.reasoning_content || '';
              const content = data.choices[0].delta.content || '';

              if (SHOW_REASONING) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }

                if (content && reasoningStarted && !thinkingTagClosed) {
                  combinedContent = '</think>\n\n' + content;
                  thinkingTagClosed = true;
                } else if (content) {
                  combinedContent = content;
                }

                data.choices[0].delta.content = combinedContent || '';
                delete data.choices[0].delta.reasoning_content;
              } else {
                data.choices[0].delta.content = content || '';
                delete data.choices[0].delta.reasoning_content;
              }
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message?.role || 'assistant',
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy Error Context Details:', {
      message: error.message,
      status: error.response?.status,
      data: error.response?.data
    });

    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.error?.message || error.response?.data?.detail || error.message || 'Internal proxy integration error',
        type: 'proxy_execution_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
});
