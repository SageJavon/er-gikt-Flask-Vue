"""
模型
"""
import torch
from torch.nn import Module, Embedding, Linear, ModuleList, Dropout, LSTMCell
from params import DEVICE

class GIKT(Module):

    def __init__(self, num_question, num_skill, q_neighbors, s_neighbors, qs_table, agg_hops=3, emb_dim=100,
                 dropout=(0.2, 0.4), hard_recap=True, rank_k=10, pre_train=False):
        super(GIKT, self).__init__()
        self.model_name = "gikt"
        self.num_question = num_question # 题目数
        self.num_skill = num_skill # 知识点数
        self.q_neighbors = q_neighbors
        self.s_neighbors = s_neighbors
        self.agg_hops = agg_hops
        # 用于指定图神经网络（GNN）中聚合（Aggregation）的跳数。
        # 它决定了在图中从一个节点传播到其邻居的层数。这里是3
        self.qs_table = qs_table
        self.emb_dim = emb_dim # 定义emb_dim，这里是100
        self.hard_recap = hard_recap # 是否硬选择 True
        self.rank_k = rank_k # 定义：一个超参数，用于指定选择历史状态的数量。
        # 作用：在软选择历史问题时，rank_k 用于选择与当前问题最相关的 rank_k 个历史状态。
        # 使用：在代码中通过 torch.topk 函数来实现，选择得分最高的 rank_k 个历史状态。

        if pre_train:
            # 使用预训练之后的向量
            _weight_q = torch.load(f='data/q_embedding.pt') # 加载q预训练问题嵌入权重
            _weight_s = torch.load(f='data/s_embedding.pt') # 加载预训练技能嵌入权重
            self.emb_table_question = Embedding(num_question, emb_dim, _weight=_weight_q) # 问题嵌入表,多出一个用于索引为0的[空白问题]，[num_question, emb_dim]
            self.emb_table_skill = Embedding(num_skill, emb_dim, _weight=_weight_s)  # 技能嵌入表, [num_skill, emb_dim]
        else:
            # 不使用预训练，随机加载向量
            self.emb_table_question = Embedding(num_question, emb_dim) # 问题嵌入表, [num_question, emb_dim]
            self.emb_table_skill = Embedding(num_skill, emb_dim) # 技能嵌入表, [num_skill, emb_dim]
        self.emb_table_response = Embedding(2, emb_dim) # 回答结果嵌入表，[2, emb_dim] 因为回答是0/1

        # self.gru1 = GRUCell(emb_dim * 2, emb_dim) # 使用GRU网络
        # self.gru2 = GRUCell(emb_dim, emb_dim)
        self.lstm_cell = LSTMCell(input_size=emb_dim * 2, hidden_size=emb_dim) # 使用LSTM网络，输入维度: emb_dim * 2, 输出维度: emb_dim
        self.mlps4agg = ModuleList(Linear(emb_dim, emb_dim) for _ in range(agg_hops)) # 聚合的MLP列表
        self.MLP_AGG_last = Linear(emb_dim, emb_dim)# 最后的聚合MLP
        self.dropout_lstm = Dropout(dropout[0]) # LSTM的Dropout层
        # self.dropout_gru = Dropout(dropout[0])
        self.dropout_gnn = Dropout(dropout[1]) # GNN的Dropout层
        self.MLP_query = Linear(emb_dim, emb_dim) # Query的MLP
        self.MLP_key = Linear(emb_dim, emb_dim) # Key的MLP
        # 公式10中的W
        self.MLP_W = Linear(2 * emb_dim, 1)# 公式中的W, 输入维度: 2 * emb_dim, 输出维度: 1

    def forward(self, question, response, mask, DEVICE, additional_state=None, time_step=None):
        # question: [batch_size, seq_len]
        # response: [batch_size, seq_len]
        # mask: [batch_size, seq_len] 和question一样的形状, 表示在question中哪些索引是真正的数据(1), 哪些是补零的数据(0)
        # 每一个在forward中new出来的tensor都要.to(DEVICE)
        batch_size, seq_len = question.shape # batch_size表示多少个用户, seq_len表示每个用户最多回答了多少个问题，这里max_seq设置为200
        q_neighbor_size, s_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1] # 知识点和习题的邻居节点数量
        h1_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1)) # 初始化的隐状态向量, [batch_size, emb_dim]，没有用到？
        h2_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1))
        state_history = torch.zeros(batch_size, seq_len, self.emb_dim, device=DEVICE) # 记录每个时间步的隐状态, [batch_size, seq_len, emb_dim]
        y_hat = torch.zeros(batch_size, seq_len, device=DEVICE) # 用于存储对学生每个时间步的预测结果, [batch_size, seq_len]
        state = torch.zeros(batch_size, self.emb_dim, device=DEVICE)


        for t in range(seq_len - 1): # 第t时刻
            question_t = question[:, t]  # 当前时刻的问题, [batch_size]
            response_t = response[:, t] # 当前时刻的回答, [batch_size]
            mask_t = torch.eq(mask[:, t], torch.tensor(1)) # 当前时刻的 mask, [batch_size]，也就是response必须为1（回答了当前题目）
            emb_response_t = self.emb_table_response(response_t) # 当前回答的嵌入向量, [batch_size, emb_dim]
            # GNN获得习题的embedding
            node_neighbors = [question_t[mask_t]] # 当前节点的邻居节点列表,[自己, 第一跳节点, 第二跳节点...]
            _batch_size = len(node_neighbors[0]) # 当前的批量, 不一定是设定好的批量
            for i in range(self.agg_hops):
                nodes_current = node_neighbors[-1].reshape(-1) # 当前正在遍历的node(上次新添加的邻居)
                # nodes_current = nodes_current.reshape(-1)
                neighbor_shape = [_batch_size] + [(q_neighbor_size if j % 2 == 0 else s_neighbor_size) for j in range(i + 1)]
                # [t时刻问题数量, q_neighbor_size, s_neighbor_size, q_neighbor_size, ...]
                if i % 2 == 0: # 找知识点节点
                    node_neighbors.append(self.q_neighbors[nodes_current].reshape(neighbor_shape))
                    # [有效数据数量, q_neighbor_size, ...]
                else: # 找习题节点
                    node_neighbors.append(self.s_neighbors[nodes_current].reshape(neighbor_shape))
                    # [有效数据数量, s_neighbor_size, ...]
            emb_node_neighbor = [] # 每层邻居(问题或者知识点)的嵌入向量,形状为node_neighbor.shape + [emb_dim]
            for i, nodes in enumerate(node_neighbors):
                if i % 2 == 0: # 问题索引->问题向量
                    emb_node_neighbor.append(self.emb_table_question(nodes)) # [有效数据数量, emb_dim]
                else: # 技能索引->技能向量
                    emb_node_neighbor.append(self.emb_table_skill(nodes)) # [有效数据数量, emb_dim]
            emb0_question_t = self.aggregate(emb_node_neighbor) # [batch_size, emb_dim] 调用aggregate函数，返回该时刻聚合更新过的问题向量，存储在emb0_q_t
            emb_question_t = torch.zeros(batch_size, self.emb_dim, device=DEVICE) # [batch_size, emb_dim] 先初始化一个全0的向量
            emb_question_t[mask_t] = emb0_question_t # 将有效数据的位置替换为更新过的向量
            emb_question_t[~mask_t] = self.emb_table_question(question_t[~mask_t]) # 补零位置用原始问题向量

            # LSTM/GRU更新知识状态
            # gru1_input = torch.cat((emb_question_t, emb_response_t), dim=1) # [batch_size, emb_dim * 2]
            # h1_pre = self.dropout_gru(self.gru1(gru1_input, h1_pre))
            # gru2_output = self.dropout_gru(self.gru2(h1_pre, h2_pre))
            lstm_input = torch.cat((emb_question_t, emb_response_t), dim=1) # [batch_size, emb_dim * 2]
            lstm_output = self.dropout_lstm(self.lstm_cell(lstm_input)[0]) # [batch_size, emb_dim]
            '''
            lstm_output:这是在当前时刻 t 通过 LSTM 更新)的嵌入向量，表示学生在当前时刻对相关知识点的掌握程度
            state_history:这是一个三维张量，记录了每个时]步的 lstm_output，表示学生在整个序列中的历史知识状态
            emb_node_neighbor:每层邻居(问题或者知识点)的嵌入向量，表示在图神经网络(GNN)中通过聚合得到的知识点嵌入向量
            emb_question_t:这是在每个时间步t从问题嵌入self.emb_table_question 中查找得到的问题嵌入向量，表示学生在当前时间步t对问题的掌握情况
            qs_concat:这是在预测时将习题和对应知识点的嵌向量拼接起来，用于预测学生在下一个时间步对问题的掌握情况
            y_hat:这是模型的输出，表示预测的每个时间步的正确率，反映学生对题目的掌握程度。
            '''
            # 找t+1时刻的[习题]以及[其对应的知识点]
            q_next = question[:, t + 1] # [batch_size, ]
            skills_related = self.qs_table[q_next] # [batch_size, num_skill]
            skills_related_list = [] # [[num_skill1, emb_dim], [num_skill2, emb_dim], ...]
            max_num_skill = 1 # 求一个问题的最多关联的技能的数量
            for i in range(batch_size):
                skills_index = torch.nonzero(skills_related[i]).squeeze()
                if len(skills_index.shape) == 0: # 只有一个技能
                    skills_related_list.append(torch.unsqueeze(self.emb_table_skill(skills_index), dim=0)) # [1, emb_dim]
                else: # 不止一个技能
                    skills_related_list.append(self.emb_table_skill(skills_index)) # [num_skill, emb_dim]
                    if skills_index.shape[0] > max_num_skill:
                        max_num_skill = skills_index.shape[0]

            # 将习题和对应知识点embedding拼接起来
            emb_q_next = self.emb_table_question(q_next) # [batch_size, emd_dim]
            qs_concat = torch.zeros(batch_size, max_num_skill + 1, self.emb_dim).to(DEVICE) # [batch_size, max_num_skill + 1, emb_dim]
            for i, emb_skills in enumerate(skills_related_list): # emb_skills: [num_skill, emb_dim]
                num_qs = 1 + emb_skills.shape[0] # 总长度为1(问题嵌入长度) + num_skill(技能嵌入长度)
                emb_next = torch.unsqueeze(emb_q_next[i], dim=0) # [1, emb_dim]
                qs_concat[i, 0 : num_qs] = torch.cat((emb_next, emb_skills), dim=0)  # [num_qs, emb_dim]
            # 第一个问题, 无需寻找历史问题, 直接预测
            if t == 0:
                y_hat[:, 0] = 0.5 # 第一个问题默认0.5的正确率
                y_hat[:, 1] = self.predict(qs_concat, torch.unsqueeze(lstm_output, dim=1)) # 后续调用predict函数,根据当前和历史状态进行预测，并将结果存储在y_hat中
                continue
            # recap硬选择历史问题
            if self.hard_recap:
                history_time = self.recap_hard(q_next, question[:, 0:t]) # 选取哪些时刻的问题
                selected_states = [] # 不同时刻t选择的历史状态
                max_num_states = 1 # 求最大的历史状态数量
                for row, selected_time in enumerate(history_time): #遍历每个用户，和它的与当前时刻相关问题的t
                    current_state = torch.unsqueeze(lstm_output[row], dim=0) # [1, emb_dim]
                    if len(selected_time) == 0: # 没有历史状态,直接取当前状态
                        selected_states.append(current_state)
                    else: # 有历史状态,将历史状态和当前状态连接起来
                        selected_state = state_history[row, torch.tensor(selected_time, dtype=torch.int64)] # [num_selected_time, emb_dim]
                        selected_states.append(torch.cat((current_state, selected_state), dim=0)) # [num_selected_time + 1, emb_dim]
                        if (selected_state.shape[0] + 1) > max_num_states:
                            max_num_states = selected_state.shape[0] + 1
                current_history_state = torch.zeros(batch_size, max_num_states, self.emb_dim).to(DEVICE)
                # 当前状态
                for b, c_h_state in enumerate(selected_states):
                    num_states = c_h_state.shape[0]
                    current_history_state[b, 0 : num_states] = c_h_state
            else: # 软选择
                current_state = lstm_output.unsqueeze(dim=1)
                if t <= self.rank_k:
                    current_history_state = torch.cat((current_state, state_history[:, 0:t]), dim=1)
                else:
                    Q = self.emb_table_question(q_next).clone().detach().unsqueeze(dim=-1) # [batch_size, emb_dim, 1]
                    K = self.emb_table_question(question[:, 0:t]).clone().detach() # [batch_size, t, emb_dim]
                    product_score = torch.bmm(K, Q).squeeze(dim=-1) # [batch_size, t]
                    _, indices = torch.topk(product_score, k=self.rank_k, dim=1)  # [batch_size, rank_k]
                    select_history = torch.cat(tuple(state_history[i][indices[i]].unsqueeze(dim=0)
                                                     for i in range(batch_size)), dim=0) # [batch_size, rank_k, emb_dim]
                    current_history_state = torch.cat((current_state, select_history), dim=1)  # [batch_size, rank_k + 1, emb_dim]
            

            y_hat[:, t + 1] = self.predict(qs_concat, current_history_state) # 调用predict函数,根据当前和历史状态进行预测，并将结果存储在y_hat中
            h2_pre = lstm_output

            if t==time_step and additional_state != None:
                state_history[:, t] = additional_state # [batch_size,emb_dim]
            else:
                state_history[:, t] = lstm_output
            state = lstm_output # state 存储的内容应该还有一些问题；state目前返回的是seq_len长度时候的知识状态，我们想要的是time_step时刻的知识状态
        return y_hat, state

            # state_history[:, t] = lstm_output  # [batch_size, emb_dim]
        # return y_hat


    def aggregate(self, emb_node_neighbor):
        # 图扩散模型
        # 输入是节点（习题节点）的embedding，计算步骤是：将节点和邻居的embedding相加，再通过一个MLP输出（embedding维度不变），激活函数用的tanh
        # 假设聚合3跳，那么输入是[0,1,2,3]，分别表示输入节点，1跳节点，2跳节点，3跳节点，总共聚合3次
        # 第1次聚合（每次聚合使用相同的MLP），(0,1)聚合得到新的embedding，放到输入位置0上；然后(1,2)聚合得到新的embedding，放到输入位置1上；然后(2,3)聚合得到新的embedding，放到输入位置2上
        # 第2次聚合，(0',1')，聚合得到新的embedding，放到输入位置0上；然后(1',2')聚合得到新的embedding，放到输入位置1上
        # 第3次聚合，(0'',1'')，聚合得到新的embedding，放到输入位置0上
        # 最后0'''通过一个MLP得到最终的embedding
        # aggregate from outside to inside
        for i in range(self.agg_hops):
            for j in range(self.agg_hops - i):
                emb_node_neighbor[j] = self.sum_aggregate(emb_node_neighbor[j], emb_node_neighbor[j + 1], j)
        return torch.tanh(self.MLP_AGG_last(emb_node_neighbor[0])) # 返回自身的向量（索引为0）

    def sum_aggregate(self, emb_self, emb_neighbor, hop):
        # 求和式聚合, 将邻居节点求和平均之后与自己相加, 得到聚合后的特征
        emb_sum_neighbor = torch.mean(emb_neighbor, dim=-2) # [有效数据数量, emb_dim]
        emb_sum = emb_sum_neighbor + emb_self # [有效数据数量, emb_dim]
        return torch.tanh(self.dropout_gnn(self.mlps4agg[hop](emb_sum)))

    def recap_hard(self, q_next, q_history):
        # 硬选择, 直接在q_history中选出与q_next有相同技能的问题
        # q_next: [batch_size, 1], q_history: [batch_size, t-1]
        batch_size = q_next.shape[0]
        q_neighbor_size, s_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1]
        q_next = q_next.reshape(-1)
        skill_related = self.q_neighbors[q_next].reshape((batch_size, q_neighbor_size)).reshape(-1)
        q_related = self.s_neighbors[skill_related].reshape((batch_size, q_neighbor_size * s_neighbor_size)).tolist()
        time_select = [[] for _ in range(batch_size)]
        for row in range(batch_size): # 每个用户
            key = q_history[row].tolist() # 该用户的回答过的问题列表
            query = q_related[row] # 与该用户的当前问题相关的问题列表
            for t, k in enumerate(key):
                if k in query:
                    time_select[row].append(t)
        return time_select

    def recap_soft(self, rank_k=10):
        # 软选择
        pass

    def predict(self, qs_concat, current_history_state):
        # qs_concat: [batch_size, num_qs, dim_emb]
        # current_history_state: [batch_size, num_state, dim_emb]
        output_g = torch.bmm(qs_concat, torch.transpose(current_history_state, 1, 2))  # [batch_size, num_qs, num_state]
        num_qs, num_state = qs_concat.shape[1], current_history_state.shape[1]
        states = torch.unsqueeze(current_history_state, dim=1)  # [batch_size, 1, num_state, dim_emb]
        states = states.repeat(1, num_qs, 1, 1)  # [batch_size, num_qs, num_state, dim_emb]
        qs_concat2 = torch.unsqueeze(qs_concat, dim=2)  # [batch_size, num_qs, 1, dim_emb]
        qs_concat2 = qs_concat2.repeat(1, 1, num_state, 1)  # [batch_size, num_qs, num_state, dim_emb]
        K = torch.tanh(self.MLP_query(states))  # [batch_size, num_qs, num_state, dim_emb]
        Q = torch.tanh(self.MLP_key(qs_concat2))  # [batch_size, num_qs, num_state, dim_emb]
        tmp = self.MLP_W(torch.cat((Q, K), dim=-1))  # [batch_size, num_qs, num_state, 1]
        tmp = torch.squeeze(tmp, dim=-1)  # [batch_size, num_qs, num_state]
        alpha = torch.softmax(tmp, dim=2)  # [batch_size, num_qs, num_state]
        p = torch.sum(torch.sum(alpha * output_g, dim=1), dim=1)  # [batch_size, 1]
        result = torch.sigmoid(torch.squeeze(p, dim=-1)) # [batch_size, ]
        return result