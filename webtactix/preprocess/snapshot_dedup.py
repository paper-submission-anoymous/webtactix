from typing import Any, Tuple, Set, Dict, List, Optional
import re

import yaml

# ============================ 基础：深度指纹 ============================

SENTINEL = object()

def deep_signature(obj: Any, memo: Dict[int, Tuple]=None) -> Tuple:
    if memo is None:
        memo = {}
    oid = id(obj)
    if isinstance(obj, (dict, list)) and oid in memo:
        return memo[oid]

    if obj is None or isinstance(obj, (bool, int, float, str)):
        sig = ("S", type(obj).__name__, obj)
    elif isinstance(obj, dict):
        items = tuple(sorted((k, deep_signature(v, memo)) for k, v in obj.items()))
        sig = ("D", items)
    elif isinstance(obj, list):
        items = tuple(deep_signature(x, memo) for x in obj)
        sig = ("L", items)
    else:
        sig = ("X", type(obj).__name__, repr(obj))

    if isinstance(obj, (dict, list)):
        memo[oid] = sig
    return sig

def collect_all_signatures(obj: Any, sigset: Set[Tuple]=None, memo: Dict[int, Tuple]=None):
    if sigset is None:
        sigset = set()
    if memo is None:
        memo = {}
    sig = deep_signature(obj, memo)
    if sig in sigset:
        return sigset
    sigset.add(sig)
    if isinstance(obj, dict):
        for v in obj.values():
            collect_all_signatures(v, sigset, memo)
    elif isinstance(obj, list):
        for x in obj:
            collect_all_signatures(x, sigset, memo)
    return sigset

# ============================ 交互与包装器 ============================

# 不把 'cell' 放入交互集合，表头 cell 的 [idx] 由专门逻辑处理
INTERACTIVE_ROLES = {
    "link","button","textbox","combobox","listbox","searchbox",
    "radio","switch","menuitem","tab","checkbox",
    # new
    "option",
    "menuitemcheckbox","menuitemradio",
    "slider","spinbutton","scrollbar",
    "treeitem",
    "gridcell","columnheader","rowheader","group",
}



WRAPPER_KEYS = {
    "navigation","menubar","banner","contentinfo","main","menu","list","rowgroup","paragraph","tablist"
}

# 哪些交互也要参与去重（表格内仍不去重）
DEDUP_INTERACTIVE_ROLES = {"link"}
# 哪些 role 的“真正 name”可能出现在冒号右边（出于安全，仅启用 button）
NAME_AFTER_COLON_ROLES = {"button", "menuitem"}

_COLON_VALUE_RE = re.compile(r'^\s*:\s*(.+)$')

ROLE_NAME_RE = re.compile(r'^(\w+)(?:\s+"([^"]*)")?(.*)$')
def parse_role_name(k: str) -> Optional[Tuple[str, Optional[str], str]]:
    m = ROLE_NAME_RE.match(k)
    if not m:
        return None
    role, name, rest = m.group(1), m.group(2), m.group(3) or ""
    return role, name, rest

# 标签行识别（包含 cell，便于表头 cell 带 [idx] 时不被再加引号）
_LABEL_ROLES = sorted(INTERACTIVE_ROLES | {"cell"})
_LABEL_ROLE_ALT = "|".join(map(re.escape, _LABEL_ROLES))

LABEL_LINE_RE = re.compile(
    rf'^(?:{_LABEL_ROLE_ALT})\s\[\d+\](?:\s+"[^"]*")?(?:\s*\[[^\]]*\])?$'
)

CELL_OR_ROW_LINE_RE = re.compile(
    r'^(?:cell|row)$|^(?:cell|row)\s+"[^"]*"$|^(?:cell|row)\s+\(under column [^)]+\)$'
)

# 作为“键名”时也不加引号
KEY_LABEL_RE = re.compile(
    rf'^(?:{_LABEL_ROLE_ALT})\s\[\d+\](?:\s+"[^"]*")?(?:\s*\[[^\]]*\])?$'
)

KEY_CELL_OR_ROW_RE = re.compile(
    r'^(?:cell|row)(?:\s+"[^"]*")?(?:\s+\(under column [^)]+\))?$'
)

# 列表里整行 “role [idx] "name": value”
KV_LABEL_LINE_RE = re.compile(
    rf'^(?P<k>(?:{_LABEL_ROLE_ALT})\s\[\d+\](?:\s+"[^"]*")?(?:\s*\[[^\]]*\])?)\s*:\s*(?P<v>.*)$'
)

# combobox / listbox 的子项不去重
NON_DEDUP_CHILD_ROLE_IN_SELECTLIKE = {"option"}

# ============================ 表格工具 ============================

def _extract_cells_from_row(row_obj: Any) -> List[Any]:
    if not isinstance(row_obj, dict) or len(row_obj) != 1:
        return []
    (_, cells), = row_obj.items()
    return cells if isinstance(cells, list) else []

def _header_names_from_row(row_obj: Any) -> List[str]:
    cells = _extract_cells_from_row(row_obj)
    names = []
    for c in cells:
        if isinstance(c, str) and c.startswith("cell"):
            m = re.match(r'^cell(?:\s+"([^"]*)")?$', c)
            if m:
                nm = m.group(1) if m.group(1) is not None else "col1"
                names.append(nm)
            else:
                names.append("col1")
        else:
            names.append("col1")
    return names

def _annotate_cell_key(cell_key: str, header: str) -> str:
    m = re.match(r'^(cell)(?:\s+"([^"]*)")?$', cell_key)
    if not m:
        return cell_key
    head, txt = m.group(1), m.group(2)
    if txt is None:
        return f'{head} (under column {header})'
    return f'{head} "{txt} (under column {header})"'

# ===== 检测子树是否包含 table（用于去重/整棵隐藏保护父容器） =====
def contains_table(node: Any) -> bool:
    if isinstance(node, dict):
        if "table" in node:
            return True
        for v in node.values():
            if contains_table(v):
                return True
        return False
    elif isinstance(node, list):
        for el in node:
            if contains_table(el):
                return True
        return False
    return False

# ============================ 预索引（含“只表头 cell”） ============================

def _collect_header_cell_interactives(
    table_val: Any, base_path: Tuple,
    addr2idx: Dict[Tuple, int], roles: List[str], names: List[Optional[str]], nums: List[int],
    role_counts: Dict[str, int], role_name_counts: Dict[Tuple[str, str], int],
    # 新增：按 role 计数（不分 name）
    role_nums: List[int], role_global_counts: Dict[str, int],
):
    """只为 header（第一 rowgroup 的第一 row）里的 cell 建立交互索引。"""
    if not isinstance(table_val, list) or not table_val:
        return
    first_rg = table_val[0]
    if not (isinstance(first_rg, dict) and 'rowgroup' in first_rg and isinstance(first_rg['rowgroup'], list) and first_rg['rowgroup']):
        return
    header_row = first_rg['rowgroup'][0]
    cells = _extract_cells_from_row(header_row)
    for c_k, cell in enumerate(cells):
        if not (isinstance(cell, str) and cell.startswith("cell")):
            continue
        m = re.match(r'^cell(?:\s+"([^"]*)")?$', cell)
        name = m.group(1) if m else None

        # 与 render_table 内部地址规则保持一致
        addr_cell = base_path + (('D', 0), ('L', 0), ('D', 0), ('L', c_k))
        idx = len(roles)
        addr2idx[addr_cell] = idx
        roles.append("cell")
        names.append(name)

        # 旧 nums：对 cell 也维持历史策略（无 name 用 role_counts，带 name 也会+1）
        if name is None:
            num = role_counts.get("cell", 0)
            nums.append(num)
            role_counts["cell"] = num + 1
        else:
            rk = ("cell", name)
            num = role_name_counts.get(rk, 0)
            nums.append(num)
            role_name_counts[rk] = num + 1
            role_counts["cell"] = role_counts.get("cell", 0) + 1

        # 新增：role-only 次序
        rn = role_global_counts.get("cell", 0)
        role_nums.append(rn)
        role_global_counts["cell"] = rn + 1

def collect_interactives(
    node: Any, path: Tuple, addr2idx: Dict[Tuple, int],
    roles: List[str], names: List[Optional[str]], nums: List[int],
    role_counts: Dict[str, int],
    role_name_counts: Dict[Tuple[str, str], int],
    # 新增：按 role 计数（不分 name）
    role_nums: List[int], role_global_counts: Dict[str, int],
):
    # —— 仅在本函数内使用的小工具，不影响全局 ——
    def _labelish(s: Optional[str]) -> bool:
        """是否像可读标签（排除纯图标/纯空白）。"""
        if not isinstance(s, str):
            return False
        t = s.strip()
        if not t:
            return False
        return re.search(r'[A-Za-z0-9]', t) is not None

    def _maybe_choose_group_name(orig_name: Optional[str], value_text: Any, rest: str) -> Optional[str]:
        # key 里已经有 "name" 就不动
        if orig_name and orig_name.strip():
            return orig_name

        # 优先：value 是可读字符串（比如 {"group": "2025"}）
        if isinstance(value_text, str) and _labelish(value_text):
            return value_text.strip()

        # 其次：字符串形态 "group: 2025" 的冒号右边
        if isinstance(rest, str):
            rs = rest.strip()
            if rs.startswith(":"):
                cand = rs[1:].strip()
                if _labelish(cand):
                    return cand

        return orig_name

    def _maybe_choose_button_name(orig_name: Optional[str], value_text: Any, rest: str) -> Optional[str]:
        """
        针对 button：优先尝试使用 value 或 rest 冒号右侧的可读文本作为“可视 name”（用于 names[]）。
        仅当候选是 orig_name 的子串（更精炼）时才覆盖，避免误伤。
        """
        candidate = None
        if isinstance(value_text, str) and _labelish(value_text):
            candidate = value_text.strip()
        elif isinstance(rest, str):
            rs = rest.strip()
            if rs.startswith(':'):
                rs = rs[1:].strip()
                if _labelish(rs):
                    candidate = rs
        if not candidate:
            return orig_name
        if not orig_name or not isinstance(orig_name, str) or not orig_name.strip():
            return candidate
        on = orig_name.strip()
        if candidate.lower() in on.lower():
            return candidate
        return orig_name

    if isinstance(node, dict):
        items = list(node.items())
        for j, (k, v) in enumerate(items):
            addr_k = path + (('D', j),)

            # 为 table 表头 cell 建索引（dict 形态）
            if k == "table":
                _collect_header_cell_interactives(
                    v, base_path=addr_k,
                    addr2idx=addr2idx, roles=roles, names=names, nums=nums,
                    role_counts=role_counts, role_name_counts=role_name_counts,
                    role_nums=role_nums, role_global_counts=role_global_counts
                )

            pr = parse_role_name(k)
            if pr and pr[0] in INTERACTIVE_ROLES:
                role, name, rest = pr

                # 名称纠偏（仅 button）：得到最终可视 name（例如把 "per page Select" 纠正为 "Select"）
                if role == "button":
                    name = _maybe_choose_button_name(name, v, rest)

                if role == "group":
                    name = _maybe_choose_group_name(name, v, rest)

                # —— 计数按“最终可视 name”进行 ——
                name_for_count = name

                # 索引写入
                idx = len(roles)
                addr2idx[addr_k] = idx
                roles.append(role)
                names.append(name)

                # nums：无 name 按 role 计数；有 name 按 (role, name_for_count) 计数
                if name_for_count is None:
                    num = role_counts.get(role, 0)
                    nums.append(num)
                    role_counts[role] = num + 1
                else:
                    rk = (role, name_for_count)
                    num = role_name_counts.get(rk, 0)
                    nums.append(num)
                    role_name_counts[rk] = num + 1
                    role_counts[role] = role_counts.get(role, 0) + 1

                # 仅按 role 的序号（role_nums）
                rn = role_global_counts.get(role, 0)
                role_nums.append(rn)
                role_global_counts[role] = rn + 1

            # 递归子树
            collect_interactives(v, addr_k, addr2idx, roles, names, nums,
                                 role_counts, role_name_counts,
                                 role_nums, role_global_counts)

    elif isinstance(node, list):
        for i, el in enumerate(node):
            addr_el = path + (('L', i),)

            if isinstance(el, str):
                pr = parse_role_name(el)
                if pr and pr[0] in INTERACTIVE_ROLES:
                    role, name, rest = pr

                    # 名称纠偏（仅 button；整行字符串场景从 rest 的冒号右侧尝试）
                    if role == "button":
                        name = _maybe_choose_button_name(name, None, rest)

                    if role == "group":
                        name = _maybe_choose_group_name(name, None, rest)

                    name_for_count = name

                    idx = len(roles)
                    addr2idx[addr_el] = idx
                    roles.append(role)
                    names.append(name)

                    if name_for_count is None:
                        num = role_counts.get(role, 0)
                        nums.append(num)
                        role_counts[role] = num + 1
                    else:
                        rk = (role, name_for_count)
                        num = role_name_counts.get(rk, 0)
                        nums.append(num)
                        role_name_counts[rk] = num + 1
                        role_counts[role] = role_counts.get(role, 0) + 1

                    rn = role_global_counts.get(role, 0)
                    role_nums.append(rn)
                    role_global_counts[role] = rn + 1

                else:
                    collect_interactives(el, addr_el, addr2idx, roles, names, nums,
                                         role_counts, role_name_counts,
                                         role_nums, role_global_counts)

            elif isinstance(el, dict) and len(el) == 1:
                (kk, vv), = el.items()
                pr = parse_role_name(kk)
                if pr and pr[0] in INTERACTIVE_ROLES:
                    role, name, rest = pr

                    # 名称纠偏（仅 button；单键 dict 场景优先看 value vv）
                    if role == "button":
                        name = _maybe_choose_button_name(name, vv, rest)

                    if role == "group":
                        name = _maybe_choose_group_name(name, vv, rest)

                    name_for_count = name

                    idx = len(roles)
                    addr2idx[addr_el + (('D', 0),)] = idx
                    roles.append(role)
                    names.append(name)

                    if name_for_count is None:
                        num = role_counts.get(role, 0)
                        nums.append(num)
                        role_counts[role] = num + 1
                    else:
                        rk = (role, name_for_count)
                        num = role_name_counts.get(rk, 0)
                        nums.append(num)
                        role_name_counts[rk] = num + 1
                        role_counts[role] = role_counts.get(role, 0) + 1

                    rn = role_global_counts.get(role, 0)
                    role_nums.append(rn)
                    role_global_counts[role] = rn + 1

                    # 递归进入子树（路径与渲染端对齐）
                    collect_interactives(vv, addr_el + (('D', 0),),
                                         addr2idx, roles, names, nums,
                                         role_counts, role_name_counts,
                                         role_nums, role_global_counts)
                else:
                    # list 的单键 dict 非交互键：若是 table，补表头 cell 预索引；路径对齐渲染端
                    if kk == "table":
                        _collect_header_cell_interactives(
                            vv, base_path=addr_el + (('D', 0),),
                            addr2idx=addr2idx, roles=roles, names=names, nums=nums,
                            role_counts=role_counts, role_name_counts=role_name_counts,
                            role_nums=role_nums, role_global_counts=role_global_counts
                        )
                    collect_interactives(vv, addr_el + (('D', 0),),
                                         addr2idx, roles, names, nums,
                                         role_counts, role_name_counts,
                                         role_nums, role_global_counts)

            else:
                collect_interactives(el, addr_el, addr2idx, roles, names, nums,
                                     role_counts, role_name_counts,
                                     role_nums, role_global_counts)



# ============================ 标记“整棵隐藏”的子树（保护 combobox/listbox/table 后代 & 含 table 的父容器） ============================

def contains_role(node: Any, role_name: str) -> bool:
    if isinstance(node, dict):
        for k, v in node.items():
            pr = parse_role_name(k)
            if pr and pr[0] == role_name:
                return True
            if contains_role(v, role_name):
                return True
        return False
    elif isinstance(node, list):
        for el in node:
            if isinstance(el, str):
                pr = parse_role_name(el)
                if pr and pr[0] == role_name:
                    return True
            else:
                if contains_role(el, role_name):
                    return True
        return False
    else:
        return False

def mark_hidden_subtrees(node: Any, prev_sigset: Set[Tuple],
                         memo: Dict[int, Tuple]=None, out: Set[int]=None,
                         in_selectlike: bool=False, in_table: bool=False):
    """
    跨快照的“整棵隐藏”：
      - 若节点子树的深度指纹在 prev_sigset 中，会尝试隐藏；
      - 但以下情况一律不隐藏：
          * 处于 combobox/listbox/table 后代（in_selectlike/in_table 为 True）
          * 子树自身包含 combobox/listbox 或 table（保护父容器）
    """
    if memo is None: memo = {}
    if out is None: out = set()

    sig = deep_signature(node, memo)
    if sig in prev_sigset \
       and not in_selectlike and not in_table \
       and not (contains_role(node, "combobox") or contains_role(node, "listbox") or contains_table(node)):
        out.add(id(node))

    if isinstance(node, dict):
        for k, v in node.items():
            pr = parse_role_name(k)
            child_in_selectlike = in_selectlike or (pr and pr[0] in ("combobox","listbox"))
            child_in_table = in_table or (k == "table")
            mark_hidden_subtrees(v, prev_sigset, memo, out, child_in_selectlike, child_in_table)
    elif isinstance(node, list):
        for el in node:
            if isinstance(el, dict) and len(el) == 1:
                (kk, vv), = el.items()
                pr = parse_role_name(kk)
                child_in_selectlike = in_selectlike or (pr and pr[0] in ("combobox","listbox"))
                child_in_table = in_table or (kk == "table")
                mark_hidden_subtrees(vv, prev_sigset, memo, out, child_in_selectlike, child_in_table)
            else:
                mark_hidden_subtrees(el, prev_sigset, memo, out, in_selectlike, in_table)
    return out


# ============================ 表格渲染（under + 截断 + 表头 cell 贴 [idx]） ============================

def render_table(table_val: Any, *, prev_sigset: Set[Tuple], hidden_ids: Set[int],
                 addr2idx: Dict[Tuple,int], skip_indices: Set[int],
                 skip_role_names: Set[Tuple[str, Optional[str]]],
                 path: Tuple, table_max_rows: int,
                 build_fn=None):
    if not isinstance(table_val, list):
        return SENTINEL

    out = []
    header = []
    if table_val:
        first_rg = table_val[0]
        if isinstance(first_rg, dict) and 'rowgroup' in first_rg and isinstance(first_rg['rowgroup'], list) and first_rg['rowgroup']:
            header_row = first_rg['rowgroup'][0]
            header = _header_names_from_row(header_row)

    header_len = len(header)

    for rg_i, rg in enumerate(table_val):
        if isinstance(rg, dict) and 'rowgroup' in rg and isinstance(rg['rowgroup'], list):
            new_rows = []
            data_rows = rg['rowgroup']

            cap = None
            if not (rg_i == 0 and len(data_rows) == 1):
                if len(data_rows) > 0:
                    cap = table_max_rows

            kept = 0
            for row_j, row in enumerate(data_rows):
                if isinstance(row, (dict, list)) and id(row) in hidden_ids:
                    continue

                cells = _extract_cells_from_row(row)
                if not cells:
                    sub = build_fn(row, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                                   addr2idx=addr2idx, skip_indices=skip_indices,
                                   skip_role_names=skip_role_names,
                                   path=path + (('L', rg_i), ('D', 0), ('L', row_j)),
                                   table_max_rows=table_max_rows,
                                   in_table=True, in_selectlike=False)
                    if sub is not SENTINEL:
                        new_rows.append(sub)
                    continue

                annotate_under = (header_len > 0 and len(cells) == header_len)

                annotated_cells = []
                for c_k, cell in enumerate(cells):
                    hname = header[c_k] if (annotate_under and c_k < len(header)) else f"col{c_k+1}"

                    if isinstance(cell, str) and cell.startswith("cell"):
                        if rg_i == 0 and row_j == 0:
                            # 只表头 cell 贴 index
                            addr_cell = path + (('D', 0), ('L', 0), ('D', 0), ('L', c_k))
                            idx = addr2idx.get(addr_cell)
                            base = _annotate_cell_key(cell, hname) if annotate_under else cell
                            if idx is not None:
                                m = re.match(r'^(cell)(?:\s+"[^"]*")?(.*)$', base)
                                if m:
                                    m2 = re.match(r'^cell\s+"([^"]*)"(.*)$', base)
                                    if m2:
                                        name_part = m2.group(1); tail = m2.group(2) or ""
                                        annotated_cells.append(f'cell [{idx}] "{name_part}"{tail}')
                                    else:
                                        rest = m.group(2) or ""
                                        annotated_cells.append(f'cell [{idx}]{rest}')
                                else:
                                    annotated_cells.append(base)
                            else:
                                annotated_cells.append(base)
                        else:
                            annotated_cells.append(_annotate_cell_key(cell, hname) if annotate_under else cell)

                    elif isinstance(cell, dict) and len(cell) == 1:
                        (ck, cv), = cell.items()
                        new_ck = _annotate_cell_key(ck, hname) if annotate_under else ck

                        sub = build_fn(
                            cv,
                            prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                            addr2idx=addr2idx, skip_indices=skip_indices,
                            skip_role_names=skip_role_names,
                            path=path + (('L', rg_i), ('D', 0), ('L', row_j), ('D', 0), ('L', c_k), ('D', 0)),
                            table_max_rows=table_max_rows,
                            in_table=True, in_selectlike=False
                        )

                        if sub is SENTINEL or sub == {} or sub == []:
                            annotated_cells.append(new_ck)
                        else:
                            annotated_cells.append({new_ck: sub})

                    else:
                        sub = build_fn(cell, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                                       addr2idx=addr2idx, skip_indices=skip_indices,
                                       skip_role_names=skip_role_names,
                                       path=path + (('L', rg_i), ('D', 0), ('L', row_j), ('D', 0), ('L', c_k)),
                                       table_max_rows=table_max_rows,
                                       in_table=True, in_selectlike=False)
                        if sub is SENTINEL:
                            continue
                        annotated_cells.append(sub)

                new_rows.append({"row": annotated_cells})

                kept += 1
                if cap is not None and kept >= cap and row_j + 1 < len(data_rows):
                    hidden = len(data_rows) - row_j - 1
                    new_rows.append(f"[There are {hidden} more rows folded below(user can see it) but for reduce tokens, You can only view it through plan <data_extraction> plan if you want to extract information from the folded rows.]")
                    break

            if new_rows:
                out.append({"rowgroup": new_rows})
        else:
            sub = build_fn(rg, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                           addr2idx=addr2idx, skip_indices=skip_indices,
                           skip_role_names=skip_role_names,
                           path=path + (('L', rg_i),),
                           table_max_rows=table_max_rows,
                           in_table=True, in_selectlike=False)
            if sub is not SENTINEL:
                out.append(sub)

    return SENTINEL if not out else out

# ============================ 渲染：去重 + 路径一致（table 内不去重；selectlike-option 不去重） ============================

def element_dup_in_prev(prev_sigset, *, key=None, val=None, s=None) -> bool:
    memo = {}
    if s is not None:
        return deep_signature(s, memo) in prev_sigset
    if key is not None:
        return deep_signature({key: val}, memo) in prev_sigset
    return False

def build_visible_tree_with_labels(
    node: Any, *, prev_sigset: Set[Tuple], hidden_ids: Set[int],
    addr2idx: Dict[Tuple, int], skip_indices: Set[int],
    skip_role_names: Set[Tuple[str, Optional[str]]],
    path: Tuple = (), table_max_rows: int = 12,
    in_table: bool = False,
    in_selectlike: bool = False
):
    if isinstance(node, (dict, list)) and id(node) in hidden_ids:
        return SENTINEL

    # ---- dict ----
    if isinstance(node, dict):
        if len(node) == 1:
            (only_k, only_v), = node.items()

            pr0 = parse_role_name(only_k)
            if only_k == "table":
                sub = render_table(
                    only_v, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                    addr2idx=addr2idx, skip_indices=skip_indices,
                    skip_role_names=skip_role_names,
                    path=path + (('D', 0),),  # 该地址与预索引 base_path 对齐
                    table_max_rows=table_max_rows,
                    build_fn=build_visible_tree_with_labels
                )
                return SENTINEL if sub is SENTINEL else {"table": sub}

            if (only_k in WRAPPER_KEYS) and not (pr0 and pr0[0] in INTERACTIVE_ROLES):
                return build_visible_tree_with_labels(
                    only_v, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                    addr2idx=addr2idx, skip_indices=skip_indices,
                    skip_role_names=skip_role_names,
                    path=path + (('D', 0),), table_max_rows=table_max_rows,
                    in_table=in_table, in_selectlike=in_selectlike
                )

            # 非包装器：表格内不去重；且**包含 table 的子树也不去重**（保护父容器）
            if not (pr0 and pr0[0] in INTERACTIVE_ROLES):
                skip_dedup = in_table or contains_table(only_v)
                if in_selectlike and not skip_dedup:
                    prk = parse_role_name(only_k)
                    if (prk and prk[0] in NON_DEDUP_CHILD_ROLE_IN_SELECTLIKE) or contains_role(only_v, "option"):
                        skip_dedup = True
                if not skip_dedup and element_dup_in_prev(prev_sigset, key=only_k, val=only_v):
                    return SENTINEL

        out_items = []
        items = list(node.items())
        for j, (k, v) in enumerate(items):
            # if k == "/url":
            #     continue

            addr_k = path + (('D', j),)

            # **无论是否有同级元素**：table 永远不去重、也不受 skip 影响
            if k == "table":
                sub = render_table(
                    v, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                    addr2idx=addr2idx, skip_indices=skip_indices,
                    skip_role_names=skip_role_names,
                    path=addr_k, table_max_rows=table_max_rows,
                    build_fn=build_visible_tree_with_labels
                )
                if sub is not SENTINEL:
                    out_items.append(("table", sub))
                continue

            pr = parse_role_name(k)

            if pr and pr[0] in INTERACTIVE_ROLES:
                role, name, rest = pr

                idx = addr2idx.get(addr_k)
                if not in_table:
                    if (role, name) in skip_role_names:
                        continue
                    if idx is not None and idx in skip_indices:
                        continue

                child_in_selectlike = in_selectlike or (role in ("combobox","listbox"))
                # ★ link 需要参与去重（表格内不去重）
                if not in_table and role in DEDUP_INTERACTIVE_ROLES:
                    if element_dup_in_prev(prev_sigset, key=k, val=v):
                        continue

                if role == "group":
                    if name is None and isinstance(v, str) and v.strip():
                        name = v.strip()
                        nv = SENTINEL
                    else:
                        nv = build_visible_tree_with_labels(
                            v, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                            addr2idx=addr2idx, skip_indices=skip_indices,
                            skip_role_names=skip_role_names,
                            path=addr_k, table_max_rows=table_max_rows,
                            in_table=in_table, in_selectlike=child_in_selectlike
                        )
                else:
                    nv = build_visible_tree_with_labels(
                        v, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                        addr2idx=addr2idx, skip_indices=skip_indices,
                        skip_role_names=skip_role_names,
                        path=addr_k, table_max_rows=table_max_rows,
                        in_table=in_table, in_selectlike=child_in_selectlike
                    )

                if role == "textbox" and isinstance(nv, str) and nv.strip():
                    # nv = nv + " (pay attention to whether the textbox needs to be empty first)"
                    pass
                label = f'{role} [{idx}]' + (f' "{name}"' if name else '') + (rest or '') if idx is not None else k
                if nv is SENTINEL or nv == {} or nv == []:
                    out_items.append((label, {}))
                else:
                    out_items.append((label, nv))
                continue

            # 非交互：表格内不去重；selectlike 中的 option 不去重；且**包含 table 的子树不去重**
            skip_dedup = in_table or contains_table(v)
            if in_selectlike and not skip_dedup:
                prk = parse_role_name(k)
                if (prk and prk[0] in NON_DEDUP_CHILD_ROLE_IN_SELECTLIKE) or contains_role(v, "option"):
                    skip_dedup = True

            if not skip_dedup and element_dup_in_prev(prev_sigset, key=k, val=v):
                continue


            nv = build_visible_tree_with_labels(
                v, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                addr2idx=addr2idx, skip_indices=skip_indices,
                skip_role_names=skip_role_names,
                path=addr_k, table_max_rows=table_max_rows,
                in_table=in_table, in_selectlike=in_selectlike
            )
            if nv is not SENTINEL:
                out_items.append((k, nv))
        return SENTINEL if not out_items else dict(out_items)

    # ---- list ----
    elif isinstance(node, list):
        out_list = []
        for i, el in enumerate(node):
            addr_el = path + (('L', i),)

            if isinstance(el, (dict, list)) and id(el) in hidden_ids:
                continue

            # listitem 剥皮
            if isinstance(el, dict) and len(el) == 1 and 'listitem' in el:
                sub = build_visible_tree_with_labels(
                    el['listitem'], prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                    addr2idx=addr2idx, skip_indices=skip_indices,
                    skip_role_names=skip_role_names,
                    path=addr_el + (('D', 0),), table_max_rows=table_max_rows,
                    in_table=in_table, in_selectlike=in_selectlike
                )
                if sub is SENTINEL:
                    continue
                if isinstance(sub, list): out_list.extend(sub)
                else: out_list.append(sub)
                continue

            if isinstance(el, dict) and len(el) == 1:
                (k, v), = el.items()
                # if k == "/url":
                #     continue
                if k == "table":
                    sub = render_table(
                        v, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                        addr2idx=addr2idx, skip_indices=skip_indices,
                        skip_role_names=skip_role_names,
                        path=addr_el + (('D', 0),), table_max_rows=table_max_rows,
                        build_fn=build_visible_tree_with_labels
                    )
                    if sub is not SENTINEL:
                        out_list.append({"table": sub})
                    continue

                pr = parse_role_name(k)
                if pr and pr[0] in INTERACTIVE_ROLES:
                    role, name, rest = pr

                    idx = addr2idx.get(addr_el + (('D', 0),))
                    if not in_table:
                        if (role, name) in skip_role_names:
                            continue
                        if idx is not None and idx in skip_indices:
                            continue

                    child_in_selectlike = in_selectlike or (role in ("combobox","listbox"))

                    nv = build_visible_tree_with_labels(
                        v, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                        addr2idx=addr2idx, skip_indices=skip_indices,
                        skip_role_names=skip_role_names,
                        path=addr_el + (('D', 0),), table_max_rows=table_max_rows,
                        in_table=in_table, in_selectlike=child_in_selectlike
                    )

                    if role == "textbox" and isinstance(nv, str) and nv.strip():
                        nv = nv + " (pay attention to whether the textbox needs to be empty first)"

                    label = f'{role} [{idx}]' + (f' "{name}"' if name else '') + (rest or '') if idx is not None else k
                    if nv is SENTINEL or nv == {} or nv == []:
                        out_list.append(label)
                    else:
                        out_list.append({label: nv})
                    continue

                # 非交互：表格内不去重；selectlike 的 option 不去重；包含 table 的子树不去重
                skip_dedup = in_table or contains_table(v)
                if in_selectlike and not skip_dedup:
                    prk = parse_role_name(k)
                    if (prk and prk[0] in NON_DEDUP_CHILD_ROLE_IN_SELECTLIKE) or contains_role(v, "option"):
                        skip_dedup = True

                if not skip_dedup and element_dup_in_prev(prev_sigset, key=k, val=v):
                    continue

                # 包装器键
                if k in WRAPPER_KEYS:
                    sub = build_visible_tree_with_labels(
                        v, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                        addr2idx=addr2idx, skip_indices=skip_indices,
                        skip_role_names=skip_role_names,
                        path=addr_el + (('D', 0),), table_max_rows=table_max_rows,
                        in_table=in_table, in_selectlike=in_selectlike
                    )
                    if sub is not SENTINEL:
                        out_list.append({k: sub})
                    continue

                # 普通键
                sub = build_visible_tree_with_labels(
                    v, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                    addr2idx=addr2idx, skip_indices=skip_indices,
                    skip_role_names=skip_role_names,
                    path=addr_el + (('D', 0),), table_max_rows=table_max_rows,
                    in_table=in_table, in_selectlike=in_selectlike
                )
                if sub is not SENTINEL:
                    out_list.append({k: sub})
                continue

            if isinstance(el, str):
                pr = parse_role_name(el)
                if pr and pr[0] in INTERACTIVE_ROLES:
                    role, name, rest = pr
                    idx = addr2idx.get(addr_el)
                    if not in_table:
                        if (role, name) in skip_role_names:
                            continue
                        if idx is not None and idx in skip_indices:
                            continue
                    # ★ link 需要参与去重（表格内不去重）
                    if not in_table and role in DEDUP_INTERACTIVE_ROLES:
                        if element_dup_in_prev(prev_sigset, s=el):
                            continue

                    if idx is None:
                        out_list.append(el)
                    else:
                        label = f'{role} [{idx}]' + (f' "{name}"' if name else '') + (rest or '')
                        out_list.append(label)
                    continue

                # selectlike 子树里的 option：不去重，直接输出
                if pr and pr[0] in NON_DEDUP_CHILD_ROLE_IN_SELECTLIKE and in_selectlike:
                    out_list.append(el)
                    continue

                if LABEL_LINE_RE.match(el) or CELL_OR_ROW_LINE_RE.match(el):
                    out_list.append(el)
                    continue

                if not in_table and not (in_selectlike and pr and pr[0] in NON_DEDUP_CHILD_ROLE_IN_SELECTLIKE):
                    if element_dup_in_prev(prev_sigset, s=el):
                        continue
                out_list.append(el)
                continue

            sub = build_visible_tree_with_labels(
                el, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
                addr2idx=addr2idx, skip_indices=skip_indices,
                skip_role_names=skip_role_names,
                path=addr_el, table_max_rows=table_max_rows,
                in_table=in_table, in_selectlike=in_selectlike
            )
            if sub is not SENTINEL:
                out_list.append(sub)

        return SENTINEL if not out_list else out_list

    # ---- 标量 ----
    else:
        return node


# ============================ YAML dumper ============================

def _typeof(x):
    return "dict" if isinstance(x, dict) else "list" if isinstance(x, list) else "scalar"

def _prep(x):
    if isinstance(x, dict): return list(x.items())
    if isinstance(x, list): return list(x)
    return x

def _is_scalar(x): return not isinstance(x, (dict, list))

def _fmt_key_raw(k: str) -> str:
    if KEY_LABEL_RE.match(k) or KEY_CELL_OR_ROW_RE.match(k):
        return k
    if any(ch in k for ch in [":","{","}","[","]","#","|"]) or k.strip()!=k:
        return '"' + k.replace('"','\\"') + '"'
    return k

def _fmt_scalar(v):
    if isinstance(v, str):
        s = v
        if s.startswith('[') and s.endswith(']') and ' rows hidden on the page' in s:
            return s
        need = (s == "" or s.strip()!=s or any(ch in s for ch in [":","#","{","}","|"]) or ("\n" in s))
        return '"' + s.replace('"','\\"') + '"' if need else s
    if v is True: return "true"
    if v is False: return "false"
    if v is None: return "null"
    return str(v)

def dump_yaml_aria(obj: Any) -> str:
    lines = []
    stack = [{"type": _typeof(obj), "items": _prep(obj), "i": 0, "indent": 0}]
    while stack:
        f = stack[-1]
        t = f["type"]
        indent = f["indent"]
        pre = "  " * indent

        if t == "dict":
            items = f["items"]
            i = f["i"]
            if i >= len(items):
                stack.pop(); continue
            k, v = items[i]; f["i"] += 1
            if _is_scalar(v):
                lines.append(f"{pre}{_fmt_key_raw(k)}: {_fmt_scalar(v)}")
            else:
                lines.append(f"{pre}{_fmt_key_raw(k)}:")
                stack.append({"type": _typeof(v), "items": _prep(v), "i": 0, "indent": indent + 1})

        elif t == "list":
            items = f["items"]
            i = f["i"]
            if i >= len(items):
                stack.pop(); continue
            v = items[i]; f["i"] += 1

            if _is_scalar(v):
                if isinstance(v, str):
                    m = KV_LABEL_LINE_RE.match(v)
                    if m:
                        k = m.group("k"); vv = m.group("v")
                        if k.startswith("textbox [") and vv.strip():
                            vv = vv + " (pay attention to whether the textbox needs to be empty first)"
                        lines.append(f"{pre}- {_fmt_key_raw(k)}: {_fmt_scalar(vv)}")
                        continue
                    if LABEL_LINE_RE.match(v) or CELL_OR_ROW_LINE_RE.match(v):
                        lines.append(f"{pre}- {v}")
                    else:
                        lines.append(f"{pre}- {_fmt_scalar(v)}")
                else:
                    lines.append(f"{pre}- {_fmt_scalar(v)}")

            elif isinstance(v, dict) and len(v) == 1:
                (kk, vv), = v.items()
                if _is_scalar(vv):
                    lines.append(f"{pre}- {_fmt_key_raw(kk)}: {_fmt_scalar(vv)}")
                else:
                    lines.append(f"{pre}- {_fmt_key_raw(kk)}:")
                    stack.append({"type": _typeof(vv), "items": _prep(vv), "i": 0, "indent": indent + 1})
            else:
                lines.append(f"{pre}-")
                stack.append({"type": _typeof(v), "items": _prep(v), "i": 0, "indent": indent + 1})

        else:
            lines.append(f"{pre}{_fmt_scalar(f['items'])}")
            stack.pop()

    return "\n".join(lines)

# ============================ 辅助：从规格生成 (role, name) 跳过集合 ============================

def _normalize_to_list(x):
    if x is None:
        return [None]
    if isinstance(x, list):
        return x if x else [None]
    return [x]

def make_skip_role_name_set(specs: Optional[List[Dict[str, Any]]]) -> Set[Tuple[str, Optional[str]]]:
    skip: Set[Tuple[str, Optional[str]]] = set()
    if not specs:
        return skip
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        roles = _normalize_to_list(spec.get("role"))
        names = _normalize_to_list(spec.get("name"))
        for r in roles:
            if r is None:
                continue
            r_str = str(r)
            for n in names:
                n_norm = None if n is None else str(n)
                skip.add((r_str, n_norm))
    return skip

# ============================ 入口 ============================

def build_snapshot_with_dedup(
    prev_data: Any, cur_data: Any, *,
    skip_indices: List[int] = None,
    skip_role_name_specs: Optional[List[Dict[str, Any]]] = None,
    table_max_rows: int = 6
):
    """
    返回：yaml_text, roles, names, nums, role_nums

    - table：永不去重，忽略 skip；仅表头 cell 交互并贴 [idx]
    - combobox/listbox：本身不去重；其后代 option 不去重（即使被容器包裹）
    - textbox：若已填值，追加提示
    - link：在表格外参与去重
    - 跨快照整棵隐藏：combobox/listbox/table 的后代不隐藏；且**包含 table 的父容器**也不隐藏

    其中：
    - roles / names / nums 为“全量索引”的原有三元组（nums：未命名按 role 计数；命名按 (role,name) 计数）
    - role_nums：与 roles 对齐的“仅按 role 计数”的序号（不区分 name）
    """
    skip_indices = set(skip_indices or [])
    skip_role_names = make_skip_role_name_set(skip_role_name_specs)

    prev_sigset = collect_all_signatures(prev_data)

    addr2idx: Dict[Tuple, int] = {}
    roles: List[str] = []
    names: List[Optional[str]] = []
    nums:  List[int] = []
    role_counts: Dict[str, int] = {}
    role_name_counts: Dict[Tuple[str, str], int] = {}

    # 新增：仅按 role 的计数
    role_nums: List[int] = []
    role_global_counts: Dict[str, int] = {}

    collect_interactives(cur_data, (), addr2idx, roles, names, nums, role_counts, role_name_counts,
                         role_nums, role_global_counts)

    hidden_ids = mark_hidden_subtrees(cur_data, prev_sigset)

    tree = build_visible_tree_with_labels(
        cur_data, prev_sigset=prev_sigset, hidden_ids=hidden_ids,
        addr2idx=addr2idx, skip_indices=skip_indices,
        skip_role_names=skip_role_names,
        path=(), table_max_rows=table_max_rows,
        in_table=False, in_selectlike=False
    )

    yaml_text = "" if (tree is SENTINEL or tree in (None, [], {})) else dump_yaml_aria(tree)
    return yaml_text, roles, names, nums, role_nums



if __name__ == "__main__":
    prev_data = [{'link "Magento Admin Panel"': [{'/url': 'http://127.0.0.1:7780/admin/admin/'}, 'img "Magento Admin Panel"']}, {'navigation': [{'menubar': [{'listitem': [{'link "\ue604 Dashboard"': [{'/url': 'http://127.0.0.1:7780/admin/admin/dashboard/'}]}]}, {'listitem': [{'link "\ue60b Sales"': [{'/url': '#'}]}]}, {'listitem': [{'menuitem "\ue608 Catalog"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue603 Customers"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue609 Marketing"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue602 Content"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue60a Reports"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue60d Stores"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue610 System"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue612 Find Partners & Extensions"': [{'/url': 'http://127.0.0.1:7780/admin/marketplace/index/'}]}]}]}]}, 'button "System Messages: 1"', {'text': '\ue623 Failed to synchronize data to the Magento Business Intelligence service.'}, {'link "Retry Synchronization"': [{'/url': 'http://127.0.0.1:7780/admin/analytics/subscription/retry/'}]}, {'banner': ['heading "Dashboard" [level=1]', {'link "\ue600 admin"': [{'/url': 'http://127.0.0.1:7780/admin/admin/system_account/index/'}]}, {'link "\ue607"': [{'/url': 'http://127.0.0.1:7780/admin/admin/notification/index/'}]}, {'text': '\ue60c'}, 'textbox " "']}, {'main': [{'text': 'Scope:'}, 'button "All Store Views"', {'link "\ue633 What is this?"': [{'/url': 'https://docs.magento.com/user-guide/configuration/scope.html'}]}, 'button "Reload Data"', {'text': "Advanced Reporting Gain new insights and take command of your business' performance, using our dynamic product, order, and customer reports tailored to your customer data."}, {'link "Go to Advanced Reporting \ue644"': [{'/url': 'http://127.0.0.1:7780/admin/analytics/reports/show/'}]}, {'text': 'Chart is disabled. To enable the chart, click'}, {'link "here"': [{'/url': 'http://127.0.0.1:7780/admin/admin/system_config/edit/section/admin/#admin_dashboard-link'}]}, {'text': '.'}, {'list': [{'listitem': [{'text': 'Revenue'}, {'strong': '$0.00'}]}, {'listitem': [{'text': 'Tax'}, {'strong': '$0.00'}]}, {'listitem': [{'text': 'Shipping'}, {'strong': '$0.00'}]}, {'listitem': [{'text': 'Quantity'}, {'strong': '0'}]}]}, {'tablist': [{'tab "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers" [expanded] [selected]': [{'link "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers"': [{'/url': '#grid_tab_ordered_products_content'}, {'text': 'Bestsellers'}]}]}, {'tab "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products"': [{'link "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products"': [{'/url': 'http://127.0.0.1:7780/admin/admin/dashboard/productsViewed/'}, {'text': 'Most Viewed Products'}]}]}, {'tab "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers"': [{'link "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers"': [{'/url': 'http://127.0.0.1:7780/admin/admin/dashboard/customersNewest/'}, {'text': 'New Customers'}]}]}, {'tab "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers"': [{'link "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers"': [{'/url': 'http://127.0.0.1:7780/admin/admin/dashboard/customersMost/'}, {'text': 'Customers'}]}]}]}, {'tabpanel "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers"': [{'table': [{'rowgroup': [{'row "Product Price Quantity"': ['cell "Product"', 'cell "Price"', 'cell "Quantity"']}]}, {'rowgroup': [{'row "Quest Lumaflex™ Band $19.00 6"': ['cell "Quest Lumaflex™ Band"', 'cell "$19.00"', 'cell "6"']}, {'row "Sprite Stasis Ball 65 cm $27.00 5"': ['cell "Sprite Stasis Ball 65 cm"', 'cell "$27.00"', 'cell "5"']}, {'row "Sprite Stasis Ball 55 cm $23.00 5"': ['cell "Sprite Stasis Ball 55 cm"', 'cell "$23.00"', 'cell "5"']}, {'row "Overnight Duffle $45.00 5"': ['cell "Overnight Duffle"', 'cell "$45.00"', 'cell "5"']}, {'row "Sprite Yoga Strap 6 foot $14.00 4"': ['cell "Sprite Yoga Strap 6 foot"', 'cell "$14.00"', 'cell "4"']}]}]}]}, {'text': 'Lifetime Sales'}, {'strong': '$0.00'}, {'text': 'Average Order'}, {'strong': '$0.00'}, {'text': 'Last Orders'}, {'table': [{'rowgroup': [{'row "Customer Items Total"': ['cell "Customer"', 'cell "Items"', 'cell "Total"']}]}, {'rowgroup': [{'row "瀚迅 李 1 $0.00"': ['cell "瀚迅 李"', 'cell "1"', 'cell "$0.00"']}, {'row "Sarah Miller 5 $0.00"': ['cell "Sarah Miller"', 'cell "5"', 'cell "$0.00"']}, {'row "Grace Nguyen 4 $0.00"': ['cell "Grace Nguyen"', 'cell "4"', 'cell "$0.00"']}, {'row "Matt Baker 3 $0.00"': ['cell "Matt Baker"', 'cell "3"', 'cell "$0.00"']}, {'row "Lily Potter 4 $188.20"': ['cell "Lily Potter"', 'cell "4"', 'cell "$188.20"']}]}]}, {'text': 'Last Search Terms'}, {'table': [{'rowgroup': [{'row "Search Term Results Uses"': ['cell "Search Term"', 'cell "Results"', 'cell "Uses"']}]}, {'rowgroup': [{'row "tanks 23 1"': ['cell "tanks"', 'cell "23"', 'cell "1"']}, {'row "nike 0 3"': ['cell "nike"', 'cell "0"', 'cell "3"']}, {'row "Joust Bag 10 4"': ['cell "Joust Bag"', 'cell "10"', 'cell "4"']}, {'row "hollister 1 19"': ['cell "hollister"', 'cell "1"', 'cell "19"']}, {'row "Antonia Racer Tank 23 2"': ['cell "Antonia Racer Tank"', 'cell "23"', 'cell "2"']}]}]}, {'text': 'Top Search Terms'}, {'table': [{'rowgroup': [{'row "Search Term Results Uses"': ['cell "Search Term"', 'cell "Results"', 'cell "Uses"']}]}, {'rowgroup': [{'row "hollister 1 19"': ['cell "hollister"', 'cell "1"', 'cell "19"']}, {'row "Joust Bag 10 4"': ['cell "Joust Bag"', 'cell "10"', 'cell "4"']}, {'row "Antonia Racer Tank 23 2"': ['cell "Antonia Racer Tank"', 'cell "23"', 'cell "2"']}, {'row "tanks 23 1"': ['cell "tanks"', 'cell "23"', 'cell "1"']}, {'row "WP10 1 1"': ['cell "WP10"', 'cell "1"', 'cell "1"']}]}]}]}, {'contentinfo': [{'paragraph': [{'link "\ue606"': [{'/url': 'http://magento.com'}]}, {'text': 'Copyright © 2025 Magento Commerce Inc. All rights reserved.'}]}, {'paragraph': [{'strong': 'Magento'}, {'text': 'ver. 2.4.6'}]}, {'link "Privacy Policy"': [{'/url': 'https://www.adobe.com/privacy/policy.html'}]}, {'text': '|'}, {'link "Account Activity"': [{'/url': 'http://127.0.0.1:7780/admin/security/session/activity/'}]}, {'text': '|'}, {'link "Report an Issue"': [{'/url': 'https://github.com/magento/magento2/issues'}]}]}]

    cur_data = [{'link "Magento Admin Panel"': [{'/url': 'http://127.0.0.1:7780/admin/admin/'}, 'img "Magento Admin Panel"']}, {'navigation': [{'menubar': [{'listitem': [{'link "\ue604 Dashboard"': [{'/url': 'http://127.0.0.1:7780/admin/admin/dashboard/'}]}]}, {'listitem': [{'link "\ue60b Sales"': [{'/url': '#'}]}]}, {'listitem': [{'menuitem "\ue608 Catalog"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue603 Customers"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue609 Marketing"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue602 Content"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue60a Reports"': [{'/url': '#'}]}, {'strong': 'Reports'}, {'link "\ue62f"': [{'/url': '#'}]}, {'menu': [{'listitem': [{'menu': [{'listitem': [{'text': 'Marketing'}, {'menu': [{'listitem': [{'link "Products in Cart"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_shopcart/product/'}]}]}, {'listitem': [{'link "Search Terms"': [{'/url': 'http://127.0.0.1:7780/admin/search/term/report/'}]}]}, {'listitem': [{'link "Abandoned Carts"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_shopcart/abandoned/'}]}]}, {'listitem': [{'link "Newsletter Problem Reports"': [{'/url': 'http://127.0.0.1:7780/admin/newsletter/problem/'}]}]}]}]}, {'listitem': [{'text': 'Reviews'}, {'menu': [{'listitem': [{'link "By Customers"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_review/customer/'}]}]}, {'listitem': [{'link "By Products"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_review/product/'}]}]}]}]}]}]}, {'listitem': [{'menu': [{'listitem': [{'text': 'Sales'}, {'menu': [{'listitem': [{'link "Orders"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_sales/sales/'}]}]}, {'listitem': [{'link "Tax"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_sales/tax/'}]}]}, {'listitem': [{'link "Invoiced"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_sales/invoiced/'}]}]}, {'listitem': [{'link "Shipping"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_sales/shipping/'}]}]}, {'listitem': [{'link "Refunds"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_sales/refunded/'}]}]}, {'listitem': [{'link "Coupons"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_sales/coupons/'}]}]}, {'listitem': [{'link "PayPal Settlement"': [{'/url': 'http://127.0.0.1:7780/admin/paypal/paypal_reports/'}]}]}, {'listitem': [{'link "Braintree Settlement"': [{'/url': 'http://127.0.0.1:7780/admin/braintree/report/'}]}]}]}]}]}]}, {'listitem': [{'menu': [{'listitem': [{'text': 'Customers'}, {'menu': [{'listitem': [{'link "Order Total"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_customer/totals/'}]}]}, {'listitem': [{'link "Order Count"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_customer/orders/'}]}]}, {'listitem': [{'link "New"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_customer/accounts/'}]}]}]}]}, {'listitem': [{'text': 'Products'}, {'menu': [{'listitem': [{'link "Views"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_product/viewed/'}]}]}, {'listitem': [{'link "Bestsellers"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_sales/bestsellers/'}]}]}, {'listitem': [{'link "Low Stock"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_product/lowstock/'}]}]}, {'listitem': [{'link "Ordered"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_product/sold/'}]}]}, {'listitem': [{'link "Downloads"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_product/downloads/'}]}]}]}]}, {'listitem': [{'text': 'Statistics'}, {'menu': [{'listitem': [{'link "Refresh Statistics"': [{'/url': 'http://127.0.0.1:7780/admin/reports/report_statistics/'}]}]}]}]}]}]}, {'listitem': [{'menu': [{'listitem': [{'text': 'Business Intelligence'}, {'menu': [{'listitem': [{'link "Advanced Reporting\ue644"': [{'/url': 'http://127.0.0.1:7780/admin/analytics/reports/show/'}]}]}, {'listitem': [{'link "BI Essentials\ue644"': [{'/url': 'http://127.0.0.1:7780/admin/analytics/biessentials/signup/'}]}]}]}]}]}]}]}]}, {'listitem': [{'link "\ue60d Stores"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue610 System"': [{'/url': '#'}]}]}, {'listitem': [{'link "\ue612 Find Partners & Extensions"': [{'/url': 'http://127.0.0.1:7780/admin/marketplace/index/'}]}]}]}]}, 'button "System Messages: 1"', {'text': '\ue623 Failed to synchronize data to the Magento Business Intelligence service.'}, {'link "Retry Synchronization"': [{'/url': 'http://127.0.0.1:7780/admin/analytics/subscription/retry/'}]}, {'banner': ['heading "Dashboard" [level=1]', {'link "\ue600 admin"': [{'/url': 'http://127.0.0.1:7780/admin/admin/system_account/index/'}]}, {'link "\ue607"': [{'/url': 'http://127.0.0.1:7780/admin/admin/notification/index/'}]}, {'text': '\ue60c'}, 'textbox: 123']}, {'main': [{'text': 'Scope:'}, 'button "All Store Views"', {'link "\ue633 What is this?"': [{'/url': 'https://docs.magento.com/user-guide/configuration/scope.html'}]}, 'button "Reload Data"', {'text': "Advanced Reporting Gain new insights and take command of your business' performance, using our dynamic product, order, and customer reports tailored to your customer data."}, {'link "Go to Advanced Reporting \ue644"': [{'/url': 'http://127.0.0.1:7780/admin/analytics/reports/show/'}]}, {'text': 'Chart is disabled. To enable the chart, click'}, {'link "here"': [{'/url': 'http://127.0.0.1:7780/admin/admin/system_config/edit/section/admin/#admin_dashboard-link'}]}, {'text': '.'}, {'list': [{'listitem': [{'text': 'Revenue'}, {'strong': '$0.00'}]}, {'listitem': [{'text': 'Tax'}, {'strong': '$0.00'}]}, {'listitem': [{'text': 'Shipping'}, {'strong': '$0.00'}]}, {'listitem': [{'text': 'Quantity'}, {'strong': '0'}]}]}, {'tablist': [{'tab "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers" [expanded] [selected]': [{'link "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers"': [{'/url': '#grid_tab_ordered_products_content'}, {'text': 'Bestsellers'}]}]}, {'tab "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products"': [{'link "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products"': [{'/url': 'http://127.0.0.1:7780/admin/admin/dashboard/productsViewed/'}, {'text': 'Most Viewed Products'}]}]}, {'tab "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers"': [{'link "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers"': [{'/url': 'http://127.0.0.1:7780/admin/admin/dashboard/customersNewest/'}, {'text': 'New Customers'}]}]}, {'tab "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers"': [{'link "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers"': [{'/url': 'http://127.0.0.1:7780/admin/admin/dashboard/customersMost/'}, {'text': 'Customers'}]}]}]}, {'tabpanel "The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers"': [{'table': [{'rowgroup': [{'row "Product Price Quantity"': ['cell "Product"', 'cell "Price"', 'cell "Quantity"']}]}, {'rowgroup': [{'row "Quest Lumaflex™ Band $19.00 6"': ['cell "Quest Lumaflex™ Band"', 'cell "$19.00"', 'cell "6"']}, {'row "Sprite Stasis Ball 65 cm $27.00 5"': ['cell "Sprite Stasis Ball 65 cm"', 'cell "$27.00"', 'cell "5"']}, {'row "Sprite Stasis Ball 55 cm $23.00 5"': ['cell "Sprite Stasis Ball 55 cm"', 'cell "$23.00"', 'cell "5"']}, {'row "Overnight Duffle $45.00 5"': ['cell "Overnight Duffle"', 'cell "$45.00"', 'cell "5"']}, {'row "Sprite Yoga Strap 6 foot $14.00 4"': ['cell "Sprite Yoga Strap 6 foot"', 'cell "$14.00"', 'cell "4"']}]}]}]}, {'text': 'Lifetime Sales'}, {'strong': '$0.00'}, {'text': 'Average Order'}, {'strong': '$0.00'}, {'text': 'Last Orders'}, {'table': [{'rowgroup': [{'row "Customer Items Total"': ['cell "Customer"', 'cell "Items"', 'cell "Total"']}]}, {'rowgroup': [{'row "瀚迅 李 1 $0.00"': ['cell "瀚迅 李"', 'cell "1"', 'cell "$0.00"']}, {'row "Sarah Miller 5 $0.00"': ['cell "Sarah Miller"', 'cell "5"', 'cell "$0.00"']}, {'row "Grace Nguyen 4 $0.00"': ['cell "Grace Nguyen"', 'cell "4"', 'cell "$0.00"']}, {'row "Matt Baker 3 $0.00"': ['cell "Matt Baker"', 'cell "3"', 'cell "$0.00"']}, {'row "Lily Potter 4 $188.20"': ['cell "Lily Potter"', 'cell "4"', 'cell "$188.20"']}]}]}, {'text': 'Last Search Terms'}, {'table': [{'rowgroup': [{'row "Search Term Results Uses"': ['cell "Search Term"', 'cell "Results"', 'cell "Uses"']}]}, {'rowgroup': [{'row "tanks 23 1"': ['cell "tanks"', 'cell "23"', 'cell "1"']}, {'row "nike 0 3"': ['cell "nike"', 'cell "0"', 'cell "3"']}, {'row "Joust Bag 10 4"': ['cell "Joust Bag"', 'cell "10"', 'cell "4"']}, {'row "hollister 1 19"': ['cell "hollister"', 'cell "1"', 'cell "19"']}, {'row "Antonia Racer Tank 23 2"': ['cell "Antonia Racer Tank"', 'cell "23"', 'cell "2"']}]}]}, {'text': 'Top Search Terms'}, {'table': [{'rowgroup': [{'row "Search Term Results Uses"': ['cell "Search Term"', 'cell "Results"', 'cell "Uses"']}]}, {'rowgroup': [{'row "hollister 1 19"': ['cell "hollister"', 'cell "1"', 'cell "19"']}, {'row "Joust Bag 10 4"': ['cell "Joust Bag"', 'cell "10"', 'cell "4"']}, {'row "Antonia Racer Tank 23 2"': ['cell "Antonia Racer Tank"', 'cell "23"', 'cell "2"']}, {'row "tanks 23 1"': ['cell "tanks"', 'cell "23"', 'cell "1"']}, {'row "WP10 1 1"': ['cell "WP10"', 'cell "1"', 'cell "1"']}]}]}]}, {'contentinfo': [{'paragraph': [{'link "\ue606"': [{'/url': 'http://magento.com'}]}, {'text': 'Copyright © 2025 Magento Commerce Inc. All rights reserved.'}]}, {'paragraph': [{'strong': 'Magento'}, {'text': 'ver. 2.4.6'}]}, {'link "Privacy Policy"': [{'/url': 'https://www.adobe.com/privacy/policy.html'}]}, {'text': '|'}, {'link "Account Activity"': [{'/url': 'http://127.0.0.1:7780/admin/security/session/activity/'}]}, {'text': '|'}, {'link "Report an Issue"': [{'/url': 'https://github.com/magento/magento2/issues'}]}]}]

    skip = []
    skip_dict = []
    yaml_text, roles, names, nums, role_nums = build_snapshot_with_dedup(
        None, cur_data, skip_indices=skip, skip_role_name_specs=skip_dict
    )
    print("===== Snapshot (with skips) =====")
    print(yaml_text)
    # print(yaml.dump(prev_data))
    print("\n===== Index (all elements, including skipped) =====")
    for i, (r, n, m, nm) in enumerate(zip(roles, names, nums, role_nums)):
        print(f"[{i}] role={r!r}, name={n!r}, num={m}, role_num={nm}")