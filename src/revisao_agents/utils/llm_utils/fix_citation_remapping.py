"""
SOLUÇÃO COMPLETA: Rastreamento de Citações com Mapeamento Fonte → URL
======================================================================

Problema: Citações [14] no texto não correspondem às URLs [1], [2], [3] nas referências
Causa: Não há rastreamento de qual URL foi usada em qual parágrafo

Solução:
1. Durante escrita: Rastrear qual URL foi usada em cada trecho
2. Durante consolidação: Mapear citações para URLs reais
3. Re-numerar tudo consistentemente

Este é um patch para escrever_secoes_node E consolidar_node
"""

import re
from typing import Dict, List, Tuple, Set


class CitacaoRastreador:
    """Rastreia qual fonte foi usada em qual parágrafo."""
    
    def __init__(self):
        # Mapa: {numero_citacao: url}
        self.citacao_para_url: Dict[int, str] = {}
        
        # Contador de fontes para esta seção
        self.fonte_counter = 1
        
        # URLs já adicionadas (evita duplicatas)
        self.urls_vistas: Set[str] = set()
    
    def adicionar_fonte(self, url: str) -> int:
        """
        Registra uma URL e retorna o número de citação.
        
        Exemplo:
            tracker.adicionar_fonte("https://paper1.pdf")
            → retorna 1
            
            tracker.adicionar_fonte("https://paper2.pdf")
            → retorna 2
            
            tracker.adicionar_fonte("https://paper1.pdf")  # duplicada
            → retorna 1 (já tinha sido adicionada)
        """
        if url in self.urls_vistas:
            # Encontra o número já atribuído
            for num, u in self.citacao_para_url.items():
                if u == url:
                    return num
        
        # URL nova
        self.urls_vistas.add(url)
        num = self.fonte_counter
        self.citacao_para_url[num] = url
        self.fonte_counter += 1
        return num
    
    def obter_urls_ordenadas(self) -> List[str]:
        """Retorna lista de URLs na ordem das citações [1], [2], [3]..."""
        urls = []
        for i in range(1, self.fonte_counter):
            if i in self.citacao_para_url:
                urls.append(self.citacao_para_url[i])
        return urls
    
    def obter_mapa_completo(self) -> Dict[int, str]:
        """Retorna dicionário completo {numero_citacao: url}"""
        return self.citacao_para_url.copy()


def extrair_citacoes_numeradas(texto: str) -> List[int]:
    """
    Extrai ALL [N] do texto na ordem de aparição.
    
    Exemplo:
        "conforme [13]... e [14]... e [13] novamente"
        → [13, 14, 13]
    """
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, texto)
    return [int(m) for m in matches]


def criar_mapa_remapeamento(citacoes_originais: List[int]) -> Dict[int, int]:
    """
    Cria mapa old_idx → new_idx na ordem de primeira aparição.
    
    Exemplo:
        [13, 14, 13, 15]
        → {13: 1, 14: 2, 15: 3}
    """
    mapa = {}
    novo_idx = 1
    for old_idx in citacoes_originais:
        if old_idx not in mapa:
            mapa[old_idx] = novo_idx
            novo_idx += 1
    return mapa


def remapear_texto_com_rastreamento(
    texto: str,
    fonte_map_original: Dict[int, str],
    mapa_remapeamento: Dict[int, int],
) -> Tuple[str, Dict[int, str], Dict[int, int]]:
    """
    Re-mapeia citações E retorna novo mapa fonte→url.
    
    Args:
        texto: "conforme [13]... e [14]..."
        fonte_map_original: {13: "url13", 14: "url14", ...}
        mapa_remapeamento: {13: 1, 14: 2, ...}
    
    Returns:
        (texto_remapeado, novo_mapa_fonte_url, mapa_remapeamento)
    
    Exemplo:
        texto = "conforme [13]... e [14]..."
        fonte_map = {13: "https://paper13.pdf", 14: "https://paper14.pdf"}
        mapa = {13: 1, 14: 2}
        
        → texto_novo = "conforme [1]... e [2]..."
        → novo_mapa = {1: "https://paper13.pdf", 2: "https://paper14.pdf"}
        → mapa = {13: 1, 14: 2}
    """
    
    def substituir(match):
        old_idx = int(match.group(1))
        new_idx = mapa_remapeamento.get(old_idx, old_idx)
        return f"[{new_idx}]"
    
    texto_remapeado = re.sub(r'\[(\d+)\]', substituir, texto)
    
    # Cria novo mapa fonte→url
    novo_mapa = {}
    for old_idx, new_idx in mapa_remapeamento.items():
        if old_idx in fonte_map_original:
            novo_mapa[new_idx] = fonte_map_original[old_idx]
    
    return texto_remapeado, novo_mapa, mapa_remapeamento


def sincronizar_texto_com_references(
    texto: str,
    fonte_map_original: Dict[int, str],
) -> Tuple[str, List[str]]:
    """
    Sincroniza completamente texto e referências.
    
    Workflow:
    1. Extrai [N] do texto
    2. Cria mapa remapeamento
    3. Re-numera [N]
    4. Re-ordena URLs
    
    Args:
        texto: "conforme [13]... e [14]..."
        fonte_map_original: {13: "url", 14: "url", ...}
    
    Returns:
        (texto_com_[1][2], urls_ordenadas_[1][2])
    """
    
    # Extrai citações originais
    citacoes = extrair_citacoes_numeradas(texto)
    
    if not citacoes:
        return texto, list(fonte_map_original.values())
    
    # Cria mapa remapeamento
    mapa = criar_mapa_remapeamento(citacoes)
    
    # Re-mapeia
    texto_novo, novo_mapa_fonte, _ = remapear_texto_com_rastreamento(
        texto, fonte_map_original, mapa
    )
    
    # Extrai URLs na ordem nova
    urls_ordenadas = []
    for i in range(1, len(novo_mapa_fonte) + 1):
        if i in novo_mapa_fonte:
            urls_ordenadas.append(novo_mapa_fonte[i])
    
    return texto_novo, urls_ordenadas

# ============================================================================
# TESTES
# ============================================================================

def teste_rastreamento_completo():
    """Teste completo do sistema."""
    
    print("=" * 70)
    print("TESTE: Rastreamento Completo de Citações")
    print("=" * 70)
    
    # Simula escrever_secoes_node
    print("\n[1] FASE: Escrita com citações [13], [14]")
    
    texto_escrito = """
    Conforme mostra [13], um estudo recente [14] demonstra que [13] é válido.
    Além disso, [15] refuta a teoria anterior [14].
    """
    
    # Simula fonte_map do corpus
    fonte_map_original = {
        13: "https://paper-chronos-1.pdf",
        14: "https://paper-chronos-2.pdf",
        15: "https://paper-lstm.pdf",
    }
    
    print(f"\n📝 Texto original (com citações [13], [14], [15]):")
    print(texto_escrito.strip())
    
    print(f"\n📚 Fonte map original:")
    for idx, url in fonte_map_original.items():
        print(f"   [{idx}] {url}")
    
    # Simula consolidar_node
    print("\n[2] CONSOLIDANDO: Sincronizando citações")
    
    texto_sincronizado, urls_ordenadas = sincronizar_texto_com_references(
        texto_escrito,
        fonte_map_original
    )
    
    print(f"\n✅ Texto sincronizado (com citações [1], [2], [3]):")
    print(texto_sincronizado.strip())
    
    print(f"\n✅ URLs ordenadas (para referências):")
    for i, url in enumerate(urls_ordenadas, 1):
        print(f"   [{i}] {url}")
    
    # Validação
    print(f"\n🔍 VALIDAÇÃO:")
    citacoes_no_texto = extrair_citacoes_numeradas(texto_sincronizado)
    print(f"   Citações encontradas: {citacoes_no_texto}")
    print(f"   URLs disponíveis: {len(urls_ordenadas)}")
    
    if max(citacoes_no_texto) <= len(urls_ordenadas):
        print(f"   ✅ SUCESSO: Todas as citações têm referência!")
    else:
        print(f"   ❌ ERRO: Citações órfãs encontradas!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    teste_rastreamento_completo()