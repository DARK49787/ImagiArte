# ImageEnhancer - IA de Desenho e Aprimoramento de Imagens

Bem-vindo ao **ImageEnhancer**, um projeto de inteligência artificial desenvolvido para transformar desenhos simples em imagens detalhadas e estilizadas no estilo Studio Ghibli. Este projeto combina modelos de aprendizado profundo como Stable Diffusion, ControlNet, BLIP-2 e Llava para processar imagens, gerar descrições automáticas e criar prompts otimizados para geração de imagens aprimoradas.

## Visão Geral

O ImageEnhancer foi criado para aprimorar imagens de forma automatizada, mantendo a fidelidade ao conteúdo original enquanto adiciona detalhes visuais no estilo Studio Ghibli. O projeto foi desenvolvido ao longo de várias semanas, com foco em corrigir erros, otimizar desempenho e experimentar diferentes modelos de IA para alcançar os melhores resultados.

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
   git clone https://github.com/<seu-usuario>/ImageEnhancer.git
   cd ImageEnhancer
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
   - Coloque o arquivo em `ComfyUI/models/checkpoints/dreamshaper_8.safetensors` (ou ajuste o caminho no script).

5. **Configure o Ollama**:
   - Instale o Ollama (instruções em [ollama.ai](https://ollama.ai)).
   - Baixe o modelo `llava:13b`:
     ```bash
     ollama pull llava:13b
     ```

## Como Executar

1. **Configure a Variável de Ambiente (para otimizar a memória da GPU)**:
   ```bash
   # No Windows:
   set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

2. **Limpe a Memória da GPU**:
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

3. **Execute o Script**:
   ```bash
   python image_to_image_gradio.py
   ```

4. **Acesse a Interface**:
   - Abra o link no terminal (geralmente `http://127.0.0.1:7860`).
   - Carregue uma imagem e clique em "Aprimorar Imagem".

## Exemplo de Uso

- **Entrada**: Uma imagem com dois retângulos laranja sobre fundo preto.
- **Saída**: Uma versão estilizada no estilo Studio Ghibli, com detalhes visuais aprimorados, iluminação suave e texturas nítidas.

## Histórico do Desenvolvimento

Entre abril e maio de 2025, trabalhei no projeto com o auxílio do meu assistente Rimuru. Aqui estão os principais marcos:

- **14 de abril**: Resolvi problemas de instalação de dependências e ajustei o script inicial pra suportar upload de imagens na minha RTX 3060.
- **16 de abril**: Melhorei a interface com uma paleta de cores futurista e implementei um modo escuro com React Native.
- **17 de abril**: Otimizei o uso de memória do script com `torch.no_grad()` e reduzi a resolução das imagens pra 512x512.
- **18 de abril**: Adicionei uma funcionalidade de login com Firebase Authentication.
- **21 de abril**: Continuei otimizando o script pra reduzir o uso de VRAM.
- **22 de abril**: Corrigi erros no `server.py`, otimizei o ambiente virtual e ajustei o suporte a CUDA.
- **23 de abril**: Ajustei a interface com React Native e Ionic pra mudar o ícone dinamicamente.
- **24 de abril**: Adicionei mais otimizações de memória com `torch.no_grad()`.
- **25 de abril**: Integramos o ControlNet pra preservar contornos e começamos a usar o `llava:13b` pra refinar prompts.
- **28 de abril**: Mudei pra Stable Diffusion XL pra suportar prompts mais longos (até 225 tokens).
- **29 de abril**: Testei o modelo `SG161222/RealVisXL_V4.0` pra mais realismo, mas voltei pro `DreamShaper 8` pra manter o estilo Studio Ghibli.
- **30 de abril**: Adicionei suporte a lineart com `LineartDetector`, mas depois voltei pro scribble pra simplificar.
- **1 de maio**: Testei o `ShermanG/ControlNet-Standard-Lineart-for-SDXL` pra lineart, mas mantive o scribble.
- **20 de maio**: Finalizei a integração do `llava:13b` pra refinar prompts e otimizei o script pra rodar eficientemente.

## Limitações

- **Memória**: Requer uma GPU com pelo menos 8 GB de VRAM. O `enable_model_cpu_offload()` ajuda, mas pode ser lento em GPUs mais fracas.
- **Tokens**: Stable Diffusion 1.5 suporta apenas 77 tokens, mas o script tá configurado pra 360 (ajustável no `refine_prompt`).
- **Estilo**: Focado no estilo Studio Ghibli. Outros estilos podem exigir ajustes nos prompts ou modelos.

## Contribuições

Contribuições são bem-vindas! Abra issues ou envie pull requests com melhorias, como novos estilos ou otimizações.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Agradecimentos

Agradeço ao Rimuru, meu assistente de IA, por me ajudar a resolver problemas e otimizar o código durante o desenvolvimento.

---

Feito com ❤️ por [Seu Nome]  
Maio de 2025