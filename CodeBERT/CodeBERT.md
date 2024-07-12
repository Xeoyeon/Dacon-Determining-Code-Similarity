# CodeBERT
'CodeBERT: A Pre-Trained Model for Programming and Natural Languages' 논문을 바탕으로 공부를 진행하였습니다.
링크 : https://arxiv.org/abs/2002.08155
<br><br>
## 요약

1. Bimodal
    
     CodeBERT는 두 가지 종류의 입력 데이터를 처리할 수 있기 때문에 "bimodal"이라고 간주. - 자연어(natural language)와 소스 코드(source code). BERT 아키텍처를 기반으로 하여 **소스 코드와 자연어 주석 간의 상관관계를 학습**
    
2. multi-layer Transformer
3. BERT의 **masked language modeling**과 ELECTRA의 **replaced token detection**을 사용하여 학습
    - Masked Language Modeling: 문장의 일부 단어를 마스킹하고 모델이 이 단어를 예측하도록 학습합니다.
    - Replaced Token Detection: 일부 토큰을 대체하고, 모델이 이 대체된 토큰이 원래 토큰과 얼마나 유사한지를 감지하도록 학습합니다
4. 6개 프로그래밍 언어(Python, Java, JavaScript, etc)를 사용하여 학습

<br><br>
## 1. Architecture

BERT, RoBERTa, Bidirectional Transformer 를 따름. RoBERTa-base 와 똑같은 구조

- BERT
    
    주변 문맥을 고려하여 무작위로 마스크 처리된 단어 시퀀스의 마스크된 단어를 예측하도록 학습
    
- RoBERTa
- Bidirectional Transformer
    - 문장을 동시에 양방향으로 처리.
    - self-attention -모든 단어 간의 관계를 한 번에 평가할 수 있어, 장거리 의존성을 효과적으로 처리. 각 입력 단어나 위치에 대해 전체 문장을 한 번에 처리하는 메커니즘
        - RNN VS Self-Attention
            
            RNN과 Self-Attention 사이의 주요 차이점은 다음과 같습니다:
            
            1. **Parallelism (병렬 처리 가능성)**:
                - **RNN**: RNN은 시퀀스의 각 단계를 순차적으로 처리해야 하므로, 각 시간 단계에서 이전 단계의 정보에 의존하여 계산을 진행합니다. 이로 인해 RNN은 한 번에 하나의 시간 단계만 병렬 처리할 수 있습니다.
                - **Self-Attention**: Self-Attention은 각 단어 간의 관계를 독립적으로 계산하기 때문에, 입력 시퀀스의 모든 위치를 동시에 처리할 수 있습니다. 이는 병렬 처리가 가능하다는 의미이며, 이로 인해 계산 효율성이 크게 향상됩니다.
            2. **Long-range Dependencies (장거리 의존성 처리)**:
                - **RNN**: RNN은 특정 시간 단계에서만 이전 시간 단계의 정보를 고려하여 다음 단계를 예측합니다. 따라서 장거리 의존성을 처리할 때 긴 문맥에서 발생할 수 있는 정보 손실이 발생할 수 있습니다.
                - **Self-Attention**: Self-Attention은 모든 단어 간의 관계를 한 번에 평가할 수 있기 때문에, 장거리 의존성을 효과적으로 처리할 수 있습니다. 각 단어는 입력 시퀀스 내의 다른 모든 단어와 직접적으로 상호작용하므로, 문장 전체의 문맥을 포괄적으로 고려할 수 있습니다.
            3. **Contextual Information (문맥 정보)**:
                - **RNN**: RNN은 각 단계에서 이전 단계의 은닉 상태를 기반으로 다음 단어의 예측을 수행합니다. 이는 단어 간의 상호작용을 제한할 수 있으며, 문맥 정보의 효과적인 모델링에 도전할 수 있습니다.
                - **Self-Attention**: Self-Attention은 모든 단어 간의 상호작용을 고려하여 각 단어의 문맥을 더욱 정교하게 모델링할 수 있습니다. 각 단어는 자신의 문맥을 계산할 때 모든 다른 단어들의 정보를 함께 고려할 수 있어, 문장의 전역적인 의미를 잘 포착할 수 있습니다.
            
            따라서 Self-Attention은 RNN과 비교하여 병렬 처리 가능성이 뛰어나며, 장거리 의존성을 더 효과적으로 처리할 수 있는 장점을 갖습니다. 이는 특히 자연어 처리와 같은 복잡한 문맥을 다루는 작업에서 Transformer 모델의 주요 강점으로 작용합니다.
            
<br><br>
## 2. Input / Output
### Training data의 특징

- bimodal data : 함수 수준의 자연어 문서와 코드가 짝을 이룸.
    
    ⇒ bimodal data (NL-PL 쌍) :  학습데이터로 사용
    
    ⇒ unimodal data (PL만을 포함하는 데이터) : generator를 만드는데 사용
    
- Function-level documentations **-** 이 함수가 어떤 작업을 수행하는지, 어떤 인자를 받고, 어떤 결과를 반환하는지 등을 설명하는 자연어 텍스트
    
    ![image](https://github.com/user-attachments/assets/46ca4c4d-3e72-4066-9e23-e032a29c13db)


    
- Explicit marker 사용 X
    
    특정 정보를 명시적으로 나타내는 표시 또는 태그. 프로그래밍 언어나 데이터 처리에서 특정 언어나 형식을 구분하기 위해 사용.
    
    ⇒ 즉, python인지 javascript인지 주석으로 명시적 구분 X
    
<br>

### input

**: 두 segment( 코드와 자연어 텍스트)의 concatenation**

```
[CLS], w1, w2, ..wn, [SEP], c1, c2, ..., cm, [EOS]
```

*자연어 텍스트는 WordPiece tokenizer를 통해 분할 :

단어를 더 작은 단위의 서브워드로 분할하여 처리.

ex) playing"이라는 단어는 "play"와 "-ing"으로 분할
<br>

### output

**: 문맥 벡터와 [CLS]가 출력됨**

1. **contextual vector representation** of each token(for both natural language and code)
    - NL : 각 단어나 서브워드는 WordPiece 토크나이저를 통해 처리되어 임베딩(embedding). Transformer 내의 self-attention 메커니즘을 사용하여 각 단어는 문장 내 다른 모든 단어와의 관계를 고려한 문맥 정보를 포함한 벡터 표현을 얻음
    - Code : 각 토큰은 프로그래밍 언어의 문법적 요소를 기반으로 분할되고 임베딩됨. 이러한 임베딩은 코드 내에서의 토큰 간의 관계를 반영하며, 코드 블록의 의미론적 표현을 포착하는 데 중요한 역할을 함.
2.  **[CLS]**
    - [CLS] 는 전체 입력 시퀀스의 요약된 표현. 이 표현은 주로 분류나 랭킹작업에서 사용되며, 모델이 입력 문장이나 코드 조각의 의미를 잘 이해하고 요약할 수 있도록 도움.
<br>
**사용한 데이터셋의 조건**

1.  각 프로젝트는 최소한 하나의 다른 프로젝트에서 사용되어야 함.
2. 텍스트 문서는 첫 번째 단락에서 자름.
3. 3개 이하의 토큰(서브워드나 단어)로 이루어진 문서는 제외
4. 3줄 미만의 함수 코드도 제외
5. 이름에 "test"라는 부분 문자열이 포함된 함수들은 제외. (테스트 목적의 함수를 배제)
    
    ex) 빨간 부분만 NL에 해당
    
    ![image](https://github.com/user-attachments/assets/a07c3917-cf4d-4dd0-95d0-caffe23aca4c)

    
<br><br>
## 3. Pre-Training

**: MLM 과 RTD**

### **Masked Language Modeling**

- NL-PL 쌍 (x ={w, c}) 을 학습에 사용
    
    **[순서]**
    
    (1),(2) : w와 c의 일부 위치를 무작위로 선택하여 마스킹
    
    (3),(4) : 선택된 위치는 특별한 [MASK] 토큰으로 대체. w와c를 각각 [MASK]로 대체한 $w_{\text{masked}}$와 $c_{\text{masked}}$를 생성. 
    
    (5) : 원래의 입력 x는 w와 c의 결합으로 표현
    
    ![image](https://github.com/user-attachments/assets/920b3aa5-9c93-463c-9a9f-0a0487b85234)

    
    *$m^w$와 $m^c$는 각각w와 c에서 마스킹할 위치를 나타냄
    
    (6) :  $m^w$와 $m^c$의 토큰을 예측
    
    ![image](https://github.com/user-attachments/assets/43a0a418-bb78-4549-993d-6005afdabf0c)


### **Replaced Token Detection**

- PL만 학습에 사용
    
   ![image](https://github.com/user-attachments/assets/046b3a72-65de-40e7-a08b-92141addf04d)

    

**[Generator]** 

2개의 데이터 생성기 : NL, PL generator 

$p_{Gw}$와  $p_{Gc}$는 무작위로 마스크된 위치에 대한 타당한 대안을 생성한다.

![image](https://github.com/user-attachments/assets/d98fae70-e423-4e41-b5f4-6a5806ca58a1)


(7),(8) : 마스킹된 위치에 대체 토큰($ŵ$,$ĉ$) 생성

(9),(10) : 원래 입력인 w와 c를 $ŵ$,$ĉ$ 로 대체(($w_{\text{corrupt}}$, $c_{\text{corrupt}}$)

(11) : 입력 x는 w와 c의 결합

**[Discriminator]**

각 단어가 원래인지 아닌지를 결정하기 위해 훈련(이진 분류). RTD의 목표는 입력의 각 위치에 대해 적용되며, 생성기가 올바른 토큰을 생성하면 해당 토큰의 레이블은 가짜(fake)가 아닌 진짜(real)

**손실 함수**

![image](https://github.com/user-attachments/assets/9afe4af8-3100-4b75-b07d-277bb28443ff)

- δ(i)는 입력 $x_{\text{corrupt}}$의 i번째 단어가 원래의 단어 $x_i$인지를 나타내는 함수. $x_{\text{corrupt}}$의 i번째 단어가 원래의 단어 $x_i$와 동일할 경우 δ(i)값은  1,  그렇지 않으면 0
- $p_{D2}$는 판별자. 입력 단어가 원래 단어일 확률을 예측하는 역할.

⇒ 생성된 입력의 각 단어가 가능한 원래의 값과 일치하도록 유도한다. 모델은 이 과정에서 잘못된 부분을 식별하고 복원함.

**전체 학습 과정**

MLM 과 RTD 손실함수를 결합한 값이 최소가 되도록 학습함.

![image](https://github.com/user-attachments/assets/003b45b7-d3ac-43e5-b173-6bb60ca7d00d)

1. Fine-Tuning 
- 자연어로 코드 검색하기(natural language code search)
    
    pre-training과 같은 방식으로 input을 주고 [CLS]를 사용하여 의미론적 언어 관계를 파악
    
- 코드를 문서화하기 (code-to-text generation)
    
    생성모델의 encoder를 CodeBERT로 initialize
    

2. NL/PL Probing
    
    선행 연구가 없으므로 직접 probing dataset을 만듦.
    
    토큰을 올바르게 예측/복원하는 능력을 테스트
    
    전문가에 의해 필터링된  candidates data를 통해 모델의 성능을 평가
    
    ⇒ 결론 :  파라미터가 고정된 상태에서, RoBERTa와 다른 code로 학습된 모델들보다 codeBERT가 뛰어난 성능을 보임
