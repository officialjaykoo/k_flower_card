# Matgo Input List

Current source:
- [topology.py](/c:/k_flower_card/DesAdapter/python/local/matgo/topology.py)

Current total:
- `157` inputs

Groups:
- `0~9`: `input_hand`
  hand raw 10 slots
- `10~91`: `input_rule_score`
  rule score/state 82 slots
- `92~102`: `input_focus_*`
  current candidate card feature 11 slots
- `103~114`: `input_board_month_bin`
  board month 12 slots
- `115~130`: `input_captured_self_*`
  self captured month/type/combo 16 slots
- `131~156`: `input_captured_opp_*`
  opp captured month/type/combo 16 slots

Index list:

```text
0. input_hand[0]
1. input_hand[1]
2. input_hand[2]
3. input_hand[3]
4. input_hand[4]
5. input_hand[5]
6. input_hand[6]
7. input_hand[7]
8. input_hand[8]
9. input_hand[9]
10. input_rule_score[0] = self_go_bonus
11. input_rule_score[1] = self_go_multiplier
12. input_rule_score[2] = self_shaking_multiplier
13. input_rule_score[3] = self_bomb_multiplier
14. input_rule_score[4] = self_bak_multiplier
15. input_rule_score[5] = self_pi_bak
16. input_rule_score[6] = self_gwang_bak
17. input_rule_score[7] = self_mong_bak
18. input_rule_score[8] = self_five_base
19. input_rule_score[9] = self_ribbon_base
20. input_rule_score[10] = self_junk_base
21. input_rule_score[11] = self_red_ribbons
22. input_rule_score[12] = self_blue_ribbons
23. input_rule_score[13] = self_plain_ribbons
24. input_rule_score[14] = self_five_birds
25. input_rule_score[15] = self_kwang
26. input_rule_score[16] = opp_go_bonus
27. input_rule_score[17] = opp_go_multiplier
28. input_rule_score[18] = opp_shaking_multiplier
29. input_rule_score[19] = opp_bomb_multiplier
30. input_rule_score[20] = opp_bak_multiplier
31. input_rule_score[21] = opp_pi_bak
32. input_rule_score[22] = opp_gwang_bak
33. input_rule_score[23] = opp_mong_bak
34. input_rule_score[24] = opp_five_base
35. input_rule_score[25] = opp_ribbon_base
36. input_rule_score[26] = opp_junk_base
37. input_rule_score[27] = opp_red_ribbons
38. input_rule_score[28] = opp_blue_ribbons
39. input_rule_score[29] = opp_plain_ribbons
40. input_rule_score[30] = opp_five_birds
41. input_rule_score[31] = opp_kwang
42. input_rule_score[32] = self_go_legal
43. input_rule_score[33] = self_stop_legal
44. input_rule_score[34] = self_go_ready
45. input_rule_score[35] = self_auto_stop_ready
46. input_rule_score[36] = self_failed_go
47. input_rule_score[37] = self_gukjin_mode_junk
48. input_rule_score[38] = self_gukjin_locked
49. input_rule_score[39] = self_pending_gukjin_choice
50. input_rule_score[40] = self_gukjin_junk_better
51. input_rule_score[41] = self_pending_president
52. input_rule_score[42] = self_pending_president_month
53. input_rule_score[43] = self_president_hold
54. input_rule_score[44] = self_president_hold_month
55. input_rule_score[45] = self_president_x4_ready
56. input_rule_score[46] = self_dokbak_risk
57. input_rule_score[47] = self_ppuk
58. input_rule_score[48] = self_jjob
59. input_rule_score[49] = self_jabbeok
60. input_rule_score[50] = self_pansseul
61. input_rule_score[51] = self_ppuk_active
62. input_rule_score[52] = self_ppuk_streak
63. input_rule_score[53] = self_held_bonus_cards
64. input_rule_score[54] = opp_go_legal
65. input_rule_score[55] = opp_stop_legal
66. input_rule_score[56] = opp_go_ready
67. input_rule_score[57] = opp_auto_stop_ready
68. input_rule_score[58] = opp_failed_go
69. input_rule_score[59] = opp_gukjin_mode_junk
70. input_rule_score[60] = opp_gukjin_locked
71. input_rule_score[61] = opp_pending_gukjin_choice
72. input_rule_score[62] = opp_gukjin_junk_better
73. input_rule_score[63] = opp_pending_president
74. input_rule_score[64] = opp_pending_president_month
75. input_rule_score[65] = opp_president_hold
76. input_rule_score[66] = opp_president_hold_month
77. input_rule_score[67] = opp_president_x4_ready
78. input_rule_score[68] = opp_dokbak_risk
79. input_rule_score[69] = opp_ppuk
80. input_rule_score[70] = opp_jjob
81. input_rule_score[71] = opp_jabbeok
82. input_rule_score[72] = opp_pansseul
83. input_rule_score[73] = opp_ppuk_active
84. input_rule_score[74] = opp_ppuk_streak
85. input_rule_score[75] = opp_held_bonus_cards
86. input_rule_score[76] = state_carry_over_multiplier
87. input_rule_score[77] = state_next_carry_over_multiplier
88. input_rule_score[78] = state_last_nagari
89. input_rule_score[79] = state_last_dokbak
90. input_rule_score[80] = state_pending_steal
91. input_rule_score[81] = state_pending_bonus_flips
92. input_focus_month[0]
93. input_focus_kwang[0]
94. input_focus_five[0]
95. input_focus_ribbon[0]
96. input_focus_junk[0]
97. input_focus_pi[0]
98. input_focus_bonus[0]
99. input_focus_red_ribbons[0]
100. input_focus_blue_ribbons[0]
101. input_focus_plain_ribbons[0]
102. input_focus_five_birds[0]
103. input_board_month_bin[0]
104. input_board_month_bin[1]
105. input_board_month_bin[2]
106. input_board_month_bin[3]
107. input_board_month_bin[4]
108. input_board_month_bin[5]
109. input_board_month_bin[6]
110. input_board_month_bin[7]
111. input_board_month_bin[8]
112. input_board_month_bin[9]
113. input_board_month_bin[10]
114. input_board_month_bin[11]
115. input_captured_self_month_bin[0]
116. input_captured_self_month_bin[1]
117. input_captured_self_month_bin[2]
118. input_captured_self_month_bin[3]
119. input_captured_self_month_bin[4]
120. input_captured_self_month_bin[5]
121. input_captured_self_month_bin[6]
122. input_captured_self_month_bin[7]
123. input_captured_self_month_bin[8]
124. input_captured_self_month_bin[9]
125. input_captured_self_month_bin[10]
126. input_captured_self_month_bin[11]
127. input_captured_self_type_bin[0] = kwang
128. input_captured_self_type_bin[1] = five
129. input_captured_self_type_bin[2] = ribbon
130. input_captured_self_type_bin[3] = pi_value
131. input_captured_self_combo_bin[0] = red_ribbons
132. input_captured_self_combo_bin[1] = blue_ribbons
133. input_captured_self_combo_bin[2] = plain_ribbons
134. input_captured_self_combo_bin[3] = five_birds
135. input_captured_self_combo_bin[4] = kwang_combo
136. input_captured_opp_month_bin[0]
137. input_captured_opp_month_bin[1]
138. input_captured_opp_month_bin[2]
139. input_captured_opp_month_bin[3]
140. input_captured_opp_month_bin[4]
141. input_captured_opp_month_bin[5]
142. input_captured_opp_month_bin[6]
143. input_captured_opp_month_bin[7]
144. input_captured_opp_month_bin[8]
145. input_captured_opp_month_bin[9]
146. input_captured_opp_month_bin[10]
147. input_captured_opp_month_bin[11]
148. input_captured_opp_type_bin[0] = kwang
149. input_captured_opp_type_bin[1] = five
150. input_captured_opp_type_bin[2] = ribbon
151. input_captured_opp_type_bin[3] = pi_value
152. input_captured_opp_combo_bin[0] = red_ribbons
153. input_captured_opp_combo_bin[1] = blue_ribbons
154. input_captured_opp_combo_bin[2] = plain_ribbons
155. input_captured_opp_combo_bin[3] = five_birds
156. input_captured_opp_combo_bin[4] = kwang_combo
```
