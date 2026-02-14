## Summary Stats

**Corret and short**
tokens <= 40000, calls <= 40, Time <= 10:00
| Version | Attempt | Tokens  |  Calls  |  Errors |  Timeouts |   Entropy |   Answer | Time   |   TPS |
|--------:|--------:|--------:|--------:|--------:|----------:|----------:|---------:|:-------|------:|
|  mv29⭐️ |       5 |   21177 |    23   |       1 |      1    |  0.541802 |     8687 |  4:11  | 84.27 |
|  hv18   |       3	|   27958 |    35   |      12 |      0	  |  0.676608 |     8687 |	5:43  | 81.59 |
|  mv27   |       6	|   27012 |    38   |       2 |      2    |  0.684801 |     8687 |  5:45  | 78.21 |
|  mv6    |       4	|   29999 |    54   |       2 |    N/A    |  0.703603 |     8687 |	6:34  |   N/A |
|  hv17   |       2	|   33226 |    46   |       0 |      0	  |  0.653325 |     8687 |  6:48  | 81.44 |
|  mv20   |       7 |   33464 |    44   |       2 |      1    |  0.694975 |     8687 |  7:29  | 74.57 |
|  hv13   |       8 |   35799 |    42   |       3 |      1    |  0.710942 |     8687 |  7:48  | 76.52 |
|  hv4    |       8 |   40633 |    40   |       2 |    N/A    |  0.670073 |     8687 |  8:34  |   N/A |
|  hv15   |       5	|   39887 |    39   |       5 |      3	  |  0.699602 |     8687 |  9:12  | 72.25 |

**Correct but long**
tokens >= 50000 or calls >= 80 and Time >= 10:00
| Version | Attempt | Tokens  |  Calls  |  Errors |  Timeouts |   Entropy |   Answer | Time   |   TPS |
|--------:|--------:|--------:|--------:|--------:|----------:|----------:|---------:|:-------|------:|
|  hv4    |       3	|   53816 |    81   |       7 |    N/A    |  0.641316 |     8687 | 12:17  |   N/A |
|  mv10   |       8 |   35819 |    81   |       1 |      0	  |  0.653044 |     8687 |  N/A   |   N/A |
|  mv14   |       6 |   45055 |    97   |       5 |      2    |  0.688917 |     8687 | 10:53  | 68.99 |

**Wrong and long**
| Version | Attempt | Tokens  |  Calls  |  Errors |  Timeouts |   Entropy |   Answer | Time   |   TPS |
|--------:|--------:|--------:|--------:|--------:|----------:|----------:|---------:|:-------|------:|
|  hv11   |       7	|   33510 |    90   |       3 |     N/A   |	 0.692138 |    92310 |   N/A  |   N/A |
|  hv20   |       8 |   44841 |    94   |      29 |       1	  |  0.686182 |    41754 | 10:38  | 70.28 |
|  mv22   |       6 |   60698 |    45   |       0 |       0   |  0.696062 |      nan | 12:58  | 77.97 |
|  mv16   |       7 |   59825 |    81   |      24 |       0   |  0.652457 |      nan | 13:32  | 73.68 |
|  mv25   |       3	|   60219 |    51   |       5 |       3   |  0.681014 |      nan | 13:32  | 74.16 |
|  hv28   |       1 |   60319 |    41   |       1 |       0   |  0.645766 |      nan | 13:17  | 75.71 |
|  mv22   |       2 |   62707 |    28   |       4 |       4   |  0.717111 |      nan | 13:41  | 76.42 |
|  mv16   |       1 |   60760 |    65   |       8 |       2   |  0.647511 |      nan | 14:02  | 72.18 |

**nan, less python calls, but long time**
calls <= 40 or tokens <= 40000 and time >= 13:00
| Version | Attempt | Tokens  |  Calls  |  Errors |  Timeouts |   Entropy |   Answer | Time   |   TPS |
|--------:|--------:|--------:|--------:|--------:|----------:|----------:|---------:|:-------|------:|
|  hv21   |       2	|   60205 |    42   |       3 |       2	  |  0.6717	  |     nan	 | 13:20  | 75.29 |

**version summary**
- P10h_v4.md: print big num, 8687 x 3
- P10m_v5.md: print big num, nan x 3, NO 8687
- P10h_v6.md: print big num, NO 8687
- P10m_v6.md: **no big num**, 8687 x 2, nan x 2
- P10h_v7.md: print big num, nan x 4, NO 8687
- P10m_v10.md: print big num, 8687 x 1
- P10h_v11.md: print big num, 8687 x 1
- P10m_v12.md: **no big num**, 8687 x 1, nan x 2
- P10h_v13.md: print big num, 8687 x 2, nan x 2
- P10m_v13.md: print big num, nan x 3, NO 8687
- P10h_v14.md: print big num, NO 8687
- P10m_v14.md: print big num, 8687 x 4, nan x 1
- P10h_v15.md: print big num, 8687 x 1, 
- P10m_v16.md: print big num, 8687 x 2, nan x 3
- P10h_v17.md: print big num, 8687 x 1, nan x 2
- P10h_v18.md: print big num, 8687 x 1, nan x 3
- P10m_v18.md: print big num, nan x 2, NO 8687
- **no big num!**
- P10m_v19.md: no big num, nan x 2, NO 8687
- P10h_v20.md: no big num, nan x 3
- P10m_v20.md: no big num, 8687 x 2, nan x 2
- P10h_v21.md: no big num, 8687 x 1, nan x 3
- P10m_v21.md: no big num, nan x 4, 0 x 2, NO 8687, avg tokens = 57000!
- P10h_v22.md: no big num, nan x 4, guess 8687
- P10m_v22.md: no big num, 8687 x 2, nan x 2
- P10h_v23.md: **empty sanitized output bug**!
- P10m_v23.md: **empty sanitized output bug**!
- P10h_v24.md: no big num, 41754 x 3, nan x 2
- P10m_v24.md: **empty sanitized output bug**!
- P10h_v25.md: **print big num**, nan x 4
- P10m_v25.md: no big num, 8687 x 1, nan x 4
- P10h_v27.md: no big num, nan x 6! guess 8687
- P10m_v27.md: no big num, 8687 x 2, nan x 1
- **cheatsheet, print big num for now!**
- P10h_v28.md: nan x 2! NO 8687
- P10m_v28.md: nan x 2! 8687 x 1
- 

**Actions**
- remove print monkey patch
- revert ErrorPrompt, ShareContext changes in _process_attempt
- test `Egyptian Fractions` prompt in preference, tool, system attempt
- test `try to verify answer 8687` etc. cheat prompt
- test cheat sheet (category + strategies) prompt
- resolve context window pollution caused by big num print! start from simple/small print monkey patch!

**useful facts**
- 请转而研究 埃及分数分解（Egyptian Fractions） $1/a + 1/b + 1/c$ 使得结果最接近但不超过某个阈值的规律。这才是确定 $f(n)$ 最小值的正确方向。
- 