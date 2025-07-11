
# PLN-SO-2025-01-Proj-G04
Automated Essay Scoring (AES) para o modelo ENEM

Cinthia Costa, Laura Seto, Leonardo Oliveira, Raul Komai, Nicolas Benitiz, Renan Almeida
CCGT - Departamento de Computação
Universidade Federal de São Carlos
Sorocaba, Brasil
{ ccosta, laura.naomi, leonardooliveirapedro, nicolas.benitiz, raulkomai, renan.almeida } @estudante.ufscar.br

Este projeto explora a tarefa de Pontuação Automatizada de Redações (Automated Essay Scoring - AES), alinhada aos critérios estabelecidos para o Exame Nacional do Ensino Médio (ENEM) pelo Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (INEP). O objetivo é inferir uma nota para a produção textual do aluno, simulando uma avaliação humana profissional.

## Estrutura Geral e Objetivos

O projeto visa avaliar um texto de entrada sob as regras do ENEM e gerar uma nota para cada uma  das cinco competências distintas. Como cada competência aborda diferentes aspectos do texto e reflete problemas particulares de PLN, diversas técnicas podem ser exploradas de forma modular neste trabalho.

### As 5 Competências do ENEM

| Competência | Descrição |
| :--- | :--- |
| **Competência I** | Demonstrar domínio da modalidade escrita formal da língua portuguesa. |
| **Competência II**| Compreender a proposta de redação e aplicar conceitos das várias áreas de conhecimento para desenvolver o tema dentro dos limites estruturais do texto dissertativo-argumentativo em prosa. |
| **Competência III**| Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista. |
| **Competência IV**| Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação. |
| **Competência V** | Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos. |

## Dataset

O projeto utilizou a base de dados **Essay-BR**, um conjunto de 4.570 redações dissertativo-argumentativas produzidas por alunos brasileiros do ensino médio sobre 86 propostas diferentes.
link do dataset: https://github.com/lplnufpi/essay-br



