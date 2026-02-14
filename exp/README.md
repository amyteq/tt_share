### Experiments & Baseline
#### python calls, errors, entropy?
[luck44a P10](./luck44a.md#p10) 的 python calls/erros 都很少, entropy 也很低, 但是正确答案(**8687**)并未出现! 
- ast_add_cache = true 导致PB drop ?? 而且对 timeout ERROR 帮助不大, P10h_v3r1.md=16, P10h_v4r1.md=15
- 貌似 temperature=1.0 & min_p=0.02 是最佳组合?? why?
- TODO: 41754 是一个经常出现的干扰项, 需要trace 8687/41754 两个答案详细的思考过程和生成的代码, 进一步考虑如何通过 prompts 或者 temperature + min_p 优化; 或者通过 plan 来优化?
- TODO: P10h_v4r1.md Attempt 1 Turn 65 就得到了正确答案 **8687**, 但一直到 Turn 114 才返回争取答案, 需要walk through!


### Submit
| base  | version   | changes               | PB    | comment                       |
|-------|-----------|-----------------------|-------|-------------------------------|
| a09   | mv2       | slice,print=True      | 41/50 | a09 mod is good ⭐️            |
| a09   | hv4       | cache=True            | 33/50 | disable it!                   |
| a09   | hv3       | temp=0,min_p=0.95     | 33/50 | try temp=1.1, min_p=0.015?    |
| a09   | mv3       | temp=1.1,min_p=0.015  | 37/50 | try temp=0.9, min_p=0.025?    |
| a09   | mv4       | temp=0.9,min_p=0.025  | 38/50 | best: temp=1.0,min_p=0.02 ⭐️  |
| a09   | mv5       | yaml (debug only)     | 36/50 | 比较py之后没有发现相差5分的原因❗️  |

### Questions
❓❓Q: 现在的 Kaggle AIMO P3 比赛, 参赛者需要使用开源 LLM (gpt-oss-120b) 的能力配合工程的方法, 使程序能正确解答并给出50道 IMO 级别的数学难题的答案 (0～99991). 目前的notebook 大体设计和关键参数实现如下:
- 基于 vllm + local jupyter sandbox + openai_harmony client 构建
- system message 包含 system_prompt 和 tool_prompt
- user message 包含 IMO problem 和 preference_prompt
- system_prompt 主要high level的规定了 LLM 的角色和general rules
- tool_prompt 主要规定了 LLM 生成python 代码时能用到的libs 和规范
- model 请求使用Stream mode, 每轮生成的response 包含python code, python代码给sandbox执行, 结果再追加到user message 继续下一轮请求, 直到 model 找到答案或者超时
- model 请求代码片段如下:
```python
stream = self.client.completions.create(
    model=self.cfg.served_model_name, 
    temperature=self.cfg.temperature, 
    logprobs=self.cfg.top_logprobs, 
    max_tokens=max_tokens, 
    prompt=prompt_ids, 
    seed=attempt_seed, 
    stream=True, 
    extra_body={
        'min_p': self.cfg.min_p, 
        'stop_token_ids': self.stop_token_ids, 
        'return_token_ids': True
    }
)
```
- 现在使用的参数: temperature=1.0, min_p=0.02. 测试了边上几组, PB 都有所下降 (41 --> 37,38)
有几个问题:
- min_p 和我们一般用的 top_p 是同一个参数吗？有何区别?
- 一般我们都用 temperature=0.0 & top_p=0.95 的参数来使用LLM解决数学问题, 为何这里用 temperature=1.0?
- 程序还有其他一些工程方面的改进, 比如对model 生成的python code 使用 AST 分析改错(末尾加print, 检查并修正slice d[:n], 加@memory.cache), 但效果都不理想, 甚至掉分, 为什么？
- 程序目前有并发多次 attempts, 并收集结果进行 score, votes 来提高正确率, 目前看下来算是能想到的比较好的工程改良之一
- 除了分阶段 (plan + solve) 给提示词引导解题外, 其他还有什么改进方法吗？
[A1.md](A1.md)

❓❓Q: 另外程序只是加了一些只在 test(debug) 下会跑的辅助代码, dump 一些配置为 yaml 文件, 其他比较过不同2次提交的python 代码, 基本都一样, 但是 PB 从 41/50 drop 到 36/50, 感觉不太合理, 程序和model 应该不会这么不问题, 会有其他原因吗？
[A2.md](A2.md)

### P10
#### h09_v3
**Attempts**
|     | Attempt | Response Length | Python Calls | Python Errors | Entropy | Answer | Time  |
|-----|---------|-----------------|--------------|---------------|---------|--------|-------|
| 0   | 2       | 24561           | 26           | 1             | 0.714   | 8687   | 4:47  |
| 1   | 3       | 24977           | 28           | 2             | 0.715   | 23     | 5:03  |
| 2   | 5       | 43889           | 66           | 6             | 0.740   | 8687   | 9:14  |
| 3   | 4       | 48118           | 57           | 4             | 0.703   | 18016  | 9:49  |
| 4   | 6       | 55281           | 62           | 5             | 0.700   | 54680  | 10:40 |
| 5   | 1       | 50393           | 57           | 6             | 0.700   | \<NA\> | 10:45 |
| 6   | 8       | 60789           | 45           | 2             | 0.704   | \<NA\> | 11:15 |
| 7   | 7       | 57299           | 69           | 5             | 0.683   | \<NA\> | 11:25 |

**Vote**
|     | Answer | Votes | Score |
|-----|--------|-------|-------|
| 0   | 8687   | 2     | 2.752 |
| 1   | 54680  | 1     | 1.429 |
| 2   | 18016  | 1     | 1.422 |
| 3   | 23     | 1     | 1.398 |

#### luck44
**Attempts**
|     | Attempt | Response Length | Python Calls | Python Errors | Entropy | Answer |
|-----|---------|-----------------|--------------|---------------|---------|--------|
| 0   | 1       | 32641           | 48           | 1             | 0.684   | 6825   |
| 1   | 8       | 31700           | 74           | 10            | 0.681   | 96985  |
| 2   | 2       | 39392           | 55           | 2             | 0.668   | 8687   |
| 3   | 5       | 41437           | 55           | 3             | 0.723   | 74234  |
| 4   | 7       | 54339           | 36           | 2             | 0.745   | 23     |
| 5   | 6       | 58980           | 55           | 2             | 0.743   | \<NA\> |
| 6   | 4       | 59324           | 37           | 5             | 0.665   | 18285  |
| 7   | 3       | 59263           | 53           | 5             | 0.722   | \<NA\> |

**Vote**
|     | Answer | Votes | Score |
|-----|--------|-------|-------|
| 0   | 18285  | 1     | 1.503 |
| 1   | 8687   | 1     | 1.498 |
| 2   | 96985  | 1     | 1.468 |
| 3   | 6825   | 1     | 1.463 |
| 4   | 74234  | 1     | 1.383 |
| 5   | 23     | 1     | 1.342 |

**h09a**
- 01271313.zip r1: final_script, temperature=1.0, min_p=0.02, 0 x 8786
- 01271324.zip r2: final_script, temperature=1.0, min_p=0.02, 0 x 8786
- 01271336.zip r3: raw_script, temperature=1.0, min_p=0.02, 2 x 8786
- 01271403.zip r5: final_script - @memory.cache, temperature=0.1, min_p=0.95, 0 x 8786
- 01271417.zip r6: raw_script, temperature=1.1, min_p=0.01, 1 x 8786

