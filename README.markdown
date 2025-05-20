## Visão Geral
O ImagiArte foi criado para aprimorar imagens de forma automatizada, mantendo a fidelidade ao conteúdo original enquanto adiciona detalhes visuais no estilo anime. O projeto foi desenvolvido ao longo de várias semanas, com foco em corrigir erros, otimizar desempenho e experimentar diferentes modelos de IA para alcançar os melhores resultados.

### Funcionalidades
- **Descrição Automática de Imagens**: Usa o modelo BLIP-2 (`Salesforce/blip2-opt-2.7b`) para gerar descrições detalhadas das imagens.
- **Refinamento de Prompts**: Utiliza o `llava:13b` do Ollama para criar prompts otimizados para Stable Diffusion, com ênfase no estilo Studio Ghibli.
- **Aprimoramento de Imagens**: Combina Stable Diffusion 1.5 (`DreamShaper 8` ou `runwayml/stable-diffusion-v1-5`) com ControlNet (`lllyasviel/control_v11p_sd15_scribble`) para gerar imagens estilizadas com contornos preservados.
- **Interface Gradio**: Interface web interativa para carregar imagens e visualizar os resultados.

## Tecnologias Utilizadas

- **Linguagem**: Python 3.10+
- **Bibliotecas**:
  - `gradio`: Interface web.
  - `torch` e `diffusers`: Modelos de aprendizado profundo e geração de imagens.
  - `transformers`: Para o modelo BLIP-2.
  - `ollama`: Integração com o modelo `llava:13b`.
  - `opencv-python` e `Pillow`: Processamento de imagens.
  - `numpy`: Manipulação de arrays.
- **Modelos**:
  - Stable Diffusion 1.5 (`DreamShaper 8` ou `runwayml/stable-diffusion-v1-5`)
  - ControlNet (`lllyasviel/control_v11p_sd15_scribble`)
  - BLIP-2 (`Salesforce/blip2-opt-2.7b`)
  - Llava (`llava:13b`)

## Como Funciona

1. **Entrada**: O usuário carrega uma imagem (ex.: dois retângulos laranja sobre fundo preto).
2. **Descrição**: O BLIP-2 gera uma descrição detalhada da imagem.
3. **Refinamento**: O `llava:13b` refina a descrição em um prompt otimizado para Stable Diffusion, mantendo o estilo Studio Ghibli.
4. **Aprimoramento**: O Stable Diffusion e o ControlNet geram uma versão aprimorada da imagem, preservando os contornos originais.
5. **Saída**: A imagem aprimorada é exibida na interface Gradio, junto com o prompt gerado.

## Pré-requisitos

- Python 3.10 ou superior.
- GPU compatível com CUDA (recomendado: NVIDIA RTX 3060 ou superior).
- Ollama instalado com o modelo `llava:13b`.
- Aproximadamente 10 GB de espaço para os modelos.

## Instalação

1. **Clone o Repositório**:
   ```bash
   git clone https://github.com/<seu-usuario>/ImagiArte.git
   cd ImagiArte
   ```

2. **Crie um Ambiente Virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as Dependências**:
   ```bash
   pip install gradio ollama Pillow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 diffusers transformers opencv-python numpy
   ```

4. **Baixe o Modelo DreamShaper 8 (Opcional)**:
   - Faça o download de `dreamshaper_8.safetensors` (ex.: no Hugging Face ou Civitai).

5. **Configure o Ollama**:
   - Instale o Ollama (instruções em [ollama.ai](https://ollama.ai)).
   - Baixe o modelo `llava:13b`:
     ```bash
     ollama pull llava:13b
     ```

## Como Executar

1. **Limpe a Memória da GPU**:
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

2. **Execute o Script**:
   ```bash
   python image_to_image_gradio.py
   ```

3. **Acesse a Interface**:
   - Abra o link no terminal (geralmente `http://127.0.0.1:7860`).
   - Carregue uma imagem e clique em "Aprimorar Imagem".

## Exemplo de Uso

- **Entrada**: Uma imagem com dois retângulos laranja sobre fundo preto.
- **Saída**: Uma versão estilizada no estilo Studio Ghibli, com detalhes visuais aprimorados, iluminação suave e texturas nítidas.

## Limitações

- **Memória**: Requer uma GPU com pelo menos 8 GB de VRAM. O `enable_model_cpu_offload()` ajuda, mas pode ser lento em GPUs mais fracas.
- **Tokens**: Stable Diffusion 1.5 suporta apenas 77 tokens, mas o script tá configurado pra 360 (ajustável no `refine_prompt`).
- **Estilo**: Focado no estilo Anime. Outros estilos podem exigir ajustes nos prompts ou modelos.

## Contribuições

Contribuições são bem-vindas! Abra issues ou envie pull requests com melhorias, como novos estilos ou otimizações.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Agradecimentos

Agradeço ao Rimuru, meu assistente de IA, por me ajudar a resolver problemas e otimizar o código durante o desenvolvimento.

---

Feito com ❤️ por [DARK49787]  
Maio de 2025