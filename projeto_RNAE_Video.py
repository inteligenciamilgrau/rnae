import math
import pygame
from typing import Tuple, List


entrada = 10

saida_desejada = 1

posicao_entrada = (0,0) # coordenada cartesiana
posicao_conexao = (4,3)

class Animation2D:
    def __init__(self, width=800, height=600, scale=40):
        """
        Inicializa a animação 2D usando Pygame
        :param width: Largura da janela
        :param height: Altura da janela
        :param scale: Fator de escala para converter coordenadas do mundo para pixels
        """
        pygame.init()
        self.width = width
        self.height = height
        self.scale = scale
        self.min_scale = 5    # Escala mínima para zoom out
        self.max_scale = 200  # Escala máxima para zoom in
        self.zoom_speed = 1.1 # Fator de multiplicação do zoom
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Rede Neural Euclidiana - Visualização")
        self.clock = pygame.time.Clock()
        
        # Cores
        self.COR_ATUAL = (0, 0, 255)      # Azul para posição atual
        self.COR_HISTORICO = (0, 255, 0)   # Verde para posições passadas
        self.COR_FUNDO = (255, 255, 255)         # Preto para o fundo
        
        # Histórico de posições
        self.historico_entrada: List[Tuple[float, float]] = []
        self.historico_conexao: List[Tuple[float, float]] = []
        
    def _converter_coordenadas(self, x: float, y: float) -> Tuple[int, int]:
        """
        Converte coordenadas do mundo para coordenadas da tela
        """
        pixel_x = int(self.width/2 + x * self.scale)
        pixel_y = int(self.height/2 - y * self.scale)  # Invertido pois y cresce para baixo na tela
        return (pixel_x, pixel_y)
    
    def adicionar_frame(self, pos_entrada: Tuple[float, float], pos_conexao: Tuple[float, float]):
        """
        Adiciona um novo frame e atualiza a visualização
        """
        self.historico_entrada.append(pos_entrada)
        self.historico_conexao.append(pos_conexao)
        self._atualizar_tela()

    def _atualizar_tela(self):
        """
        Atualiza a visualização na tela
        """
        # Limpa a tela
        self.screen.fill(self.COR_FUNDO)
        
        # Desenha o histórico
        for i in range(len(self.historico_entrada) - 1):
            pos1 = self._converter_coordenadas(*self.historico_entrada[i])
            pos2 = self._converter_coordenadas(*self.historico_conexao[i])
            pygame.draw.circle(self.screen, self.COR_HISTORICO, pos1, 3)
            pygame.draw.circle(self.screen, self.COR_HISTORICO, pos2, 3)
        
        # Desenha as posições atuais
        if self.historico_entrada:
            pos_atual_entrada = self._converter_coordenadas(*self.historico_entrada[-1])
            pos_atual_conexao = self._converter_coordenadas(*self.historico_conexao[-1])
            pygame.draw.circle(self.screen, self.COR_ATUAL, pos_atual_entrada, 5)
            pygame.draw.circle(self.screen, self.COR_ATUAL, pos_atual_conexao, 5)
            # Desenha uma linha conectando os pontos atuais
            pygame.draw.line(self.screen, self.COR_ATUAL, pos_atual_entrada, pos_atual_conexao, 1)
        
        # Desenha os eixos
        meio_x = self.width // 2
        meio_y = self.height // 2
        pygame.draw.line(self.screen, (128, 128, 128), (0, meio_y), (self.width, meio_y), 1)  # Eixo X
        pygame.draw.line(self.screen, (128, 128, 128), (meio_x, 0), (meio_x, self.height), 1) # Eixo Y
        
        pygame.display.flip()
        self.clock.tick(60)  # Limita a 60 FPS
        
    def _processar_zoom(self, event):
        """
        Processa eventos de zoom (CTRL + Scroll)
        """
        if event.type == pygame.MOUSEWHEEL and pygame.key.get_mods():
            # Zoom in (scroll up) ou zoom out (scroll down)
            zoom_in = event.y > 0
            if zoom_in:
                self.scale = min(self.scale * self.zoom_speed, self.max_scale)
            else:
                self.scale = max(self.scale / self.zoom_speed, self.min_scale)
            return True
        return False

    def manter_janela_aberta(self):
        """
        Mantém a janela aberta até que o usuário pressione ESC ou clique no X
        """
        rodando = True
        while rodando:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        rodando = False
                else:
                    # Se houve zoom, atualiza a tela
                    if self._processar_zoom(event):
                        pass
            self._atualizar_tela()
        self.encerrar()
                
    def encerrar(self):
        """
        Encerra a visualização
        """
        pygame.quit()
        
    def limpar_historico(self):
        """
        Limpa o histórico de posições
        """
        self.historico_entrada.clear()
        self.historico_conexao.clear()


erro = 10000000

class RNE():
    def distancia_euclidiana(self, ponto1, ponto2):
        """
        Calcula a distância euclidiana entre dois pontos.
        Cada ponto deve ser uma tupla ou lista de coordenadas (x, y).
        """
        return math.sqrt((ponto1[0] - ponto2[0])**2 + (ponto1[1] - ponto2[1])**2)
    
    def ativacao(self, entrada):
        """
        Ativação RELU
        """
        if entrada >= 0.1:
            return entrada
        else:
            return 0

    def ajustar_posicao_conexao(self, posicao_entrada, posicao_conexao, erro, passo=0.1):
        """
        Ajusta a posicao_conexao para mais próximo ou mais distante de posicao_entrada
        conforme o sinal do erro, com um passo proporcional ao erro.
        """
        dx = posicao_conexao[0] - posicao_entrada[0]
        dy = posicao_conexao[1] - posicao_entrada[1]
        distancia = self.distancia_euclidiana(posicao_entrada, posicao_conexao)
        if distancia == 0:
            # Evita divisão por zero
            return posicao_conexao
        # Normaliza o vetor direção
        dir_x = dx / distancia
        dir_y = dy / distancia
        # Ajusta a posição: aproxima se erro < 0, afasta se erro > 0
        novo_x = posicao_conexao[0] + passo * erro * dir_x
        novo_y = posicao_conexao[1] + passo * erro * dir_y
        return (novo_x, novo_y)

    def lei_de_coulomb(self, q1, q2, ponto1, ponto2):
        """
        Calcula a força de Coulomb entre duas cargas q1 e q2 em ponto1 e ponto2.
        Retorna o módulo da força (N).
        """
        k = 1 #8.9875517923e9  # Constante eletrostática (N·m²/C²)
        r = self.distancia_euclidiana(ponto1, ponto2)
        if r == 0:
            return float('inf')  # Evita divisão por zero
        F = k * abs(q1 * q2) / (r ** 2)
        return F

rne_V1 = RNE()
passos = 1

# Criar instância da animação
animacao = Animation2D(width=800, height=600, scale=40)  # 0.1 segundos entre frames

training = True
while training:
    print("PASSO: ", passos)
    passos+=1

    #entrada_calculada = entrada * rne_V1.distancia_euclidiana(posicao_entrada, posicao_conexao)
    entrada_calculada = entrada * rne_V1.lei_de_coulomb(1, 1, posicao_entrada, posicao_conexao)
    print("Energia entre cargas:", entrada_calculada)
    saida = rne_V1.ativacao(entrada_calculada)
    if saida_desejada is not saida:
        erro = saida - saida_desejada
        print("Erro:", erro)
    else:
        print("Erro:", 0)
        break

    if erro > 0.01 or erro < -0.01:
        # quando o erro não for zero
        # 1 - calcular o novo ponto da posicao_conexao que seja mais próximo ou mais distante que a entrada de acordo com o sinal do erro, com um passo proporcional ao erro
        novo_ponto_conexao = rne_V1.ajustar_posicao_conexao(posicao_entrada, posicao_conexao, erro)
        print(novo_ponto_conexao)
        posicao_conexao = novo_ponto_conexao
    else:
        training = False
    
    print("Saída:", saida)

    # Adicionar frame atual
    animacao.adicionar_frame(posicao_entrada, posicao_conexao)
    
# Ao final, manter a janela aberta até que o usuário feche
animacao.manter_janela_aberta()
