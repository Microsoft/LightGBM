# coding: utf-8
"""Helper script for generating config file and parameters list.

This script generates LightGBM/src/io/config_auto.cpp file
with list of all parameters, aliases table and other routines
along with parameters description in LightGBM/docs/Parameters.rst file
from the information in LightGBM/include/LightGBM/config.h file.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_parameter_infos(
    config_hpp: Path
) -> Tuple[List[Tuple[str, int]], List[List[Dict[str, List]]]]:
    """Parse config header file.

    Parameters
    ----------
    config_hpp : pathlib.Path
        Path to the config header file.

    Returns
    -------
    infos : tuple
        Tuple with names and content of sections.
    """
    is_inparameter = False
    cur_key = None
    key_lvl = 0
    cur_info: Dict[str, List] = {}
    keys = []
    member_infos: List[List[Dict[str, List]]] = []
    with open(config_hpp) as config_hpp_file:
        for line in config_hpp_file:
            if "#pragma region Parameters" in line:
                is_inparameter = True
            elif "#pragma region" in line and "Parameters" in line:
                key_lvl += 1
                cur_key = line.split("region")[1].strip()
                keys.append((cur_key, key_lvl))
                member_infos.append([])
            elif '#pragma endregion' in line:
                key_lvl -= 1
                if cur_key is not None:
                    cur_key = None
                elif is_inparameter:
                    is_inparameter = False
            elif cur_key is not None:
                line = line.strip()
                if line.startswith("//"):
                    key, _, val = line[2:].partition("=")
                    key = key.strip()
                    val = val.strip()
                    if key not in cur_info:
                        if key == "descl2" and "desc" not in cur_info:
                            cur_info["desc"] = []
                        elif key != "descl2":
                            cur_info[key] = []
                    if key == "desc":
                        cur_info["desc"].append(("l1", val))
                    elif key == "descl2":
                        cur_info["desc"].append(("l2", val))
                    else:
                        cur_info[key].append(val)
                elif line:
                    has_eqsgn = False
                    tokens = line.split("=")
                    if len(tokens) == 2:
                        if "default" not in cur_info:
                            cur_info["default"] = [tokens[1][:-1].strip()]
                        has_eqsgn = True
                    tokens = line.split()
                    cur_info["inner_type"] = [tokens[0].strip()]
                    if "name" not in cur_info:
                        if has_eqsgn:
                            cur_info["name"] = [tokens[1].strip()]
                        else:
                            cur_info["name"] = [tokens[1][:-1].strip()]
                    member_infos[-1].append(cur_info)
                    cur_info = {}

    return keys, member_infos


def get_names(
    infos: List[List[Dict[str, List]]]
) -> List[str]:
    """Get names of all parameters.

    Parameters
    ----------
    infos : list
        Content of the config header file.

    Returns
    -------
    names : list
        Names of all parameters.
    """
    names = []
    for x in infos:
        for y in x:
            names.append(y["name"][0])
    return names


def get_alias(
    infos: List[List[Dict[str, List]]]
) -> List[Tuple[str, str]]:
    """Get aliases of all parameters.

    Parameters
    ----------
    infos : list
        Content of the config header file.

    Returns
    -------
    pairs : list
        List of tuples (param alias, param name).
    """
    pairs = []
    for x in infos:
        for y in x:
            if "alias" in y:
                name = y["name"][0]
                alias = y["alias"][0].split(',')
                for name2 in alias:
                    pairs.append((name2.strip(), name))
    return pairs


def parse_check(
    check: str,
    reverse: bool = False
) -> Tuple[str, str]:
    """Parse the constraint.

    Parameters
    ----------
    check : string
        String representation of the constraint.
    reverse : bool, optional (default=False)
        Whether to reverse the sign of the constraint.

    Returns
    -------
    pair : tuple
        Parsed constraint in the form of tuple (value, sign).
    """
    try:
        idx = 1
        float(check[idx:])
    except ValueError:
        idx = 2
        float(check[idx:])
    if reverse:
        reversed_sign = {'<': '>', '>': '<', '<=': '>=', '>=': '<='}
        return check[idx:], reversed_sign[check[:idx]]
    else:
        return check[idx:], check[:idx]


def set_one_var_from_string(
    name: str,
    param_type: str,
    checks: List[str]
) -> str:
    """Construct code for auto config file for one param value.

    Parameters
    ----------
    name : string
        Name of the parameter.
    param_type : string
        Type of the parameter.
    checks : list
        Constraints of the parameter.

    Returns
    -------
    ret : string
        Lines of auto config file with getting and checks of one parameter value.
    """
    ret = ""
    univar_mapper = {"int": "GetInt", "double": "GetDouble", "bool": "GetBool", "std::string": "GetString"}
    if "vector" not in param_type:
        ret += f'  {univar_mapper[param_type]}(params, "{name}", &{name});\n'
        if len(checks) > 0:
            check_mapper = {"<": "LT", ">": "GT", "<=": "LE", ">=": "GE"}
            for check in checks:
                value, sign = parse_check(check)
                ret += f"  CHECK_{check_mapper[sign]}({name}, {value});\n"
        ret += "\n"
    else:
        ret += f'  if (GetString(params, "{name}", &tmp_str)) {{\n'
        type2 = param_type.split("<")[1][:-1]
        if type2 == "std::string":
            ret += f"    {name} = Common::Split(tmp_str.c_str(), ',');\n"
        else:
            ret += f"    {name} = Common::StringToArray<{type2}>(tmp_str, ',');\n"
        ret += "  }\n\n"
    return ret


def gen_parameter_description(
    sections: List[Tuple[str, int]],
    descriptions: List[List[Dict[str, List]]],
    params_rst: Path
) -> None:
    """Write descriptions of parameters to the documentation file.

    Parameters
    ----------
    sections : list
        Names of parameters sections.
    descriptions : list
        Structured descriptions of parameters.
    params_rst : pathlib.Path
        Path to the file with parameters documentation.
    """
    params_to_write = []
    lvl_mapper = {1: '-', 2: '~'}
    for (section_name, section_lvl), section_params in zip(sections, descriptions):
        heading_sign = lvl_mapper[section_lvl]
        params_to_write.append(f'{section_name}\n{heading_sign * len(section_name)}')
        for param_desc in section_params:
            name = param_desc['name'][0]
            default_raw = param_desc['default'][0]
            default = default_raw.strip('"') if len(default_raw.strip('"')) > 0 else default_raw
            param_type = param_desc.get('type', param_desc['inner_type'])[0].split(':')[-1].split('<')[-1].strip('>')
            options = param_desc.get('options', [])
            if len(options) > 0:
                opts = '``, ``'.join([x.strip() for x in options[0].split(',')])
                options_str = f', options: ``{opts}``'
            else:
                options_str = ''
            aliases = param_desc.get('alias', [])
            if len(aliases) > 0:
                aliases_joined = '``, ``'.join([x.strip() for x in aliases[0].split(',')])
                aliases_str = f', aliases: ``{aliases_joined}``'
            else:
                aliases_str = ''
            checks = sorted(param_desc.get('check', []))
            checks_len = len(checks)
            if checks_len > 1:
                number1, sign1 = parse_check(checks[0])
                number2, sign2 = parse_check(checks[1], reverse=True)
                checks_str = f', constraints: ``{number2} {sign2} {name} {sign1} {number1}``'
            elif checks_len == 1:
                number, sign = parse_check(checks[0])
                checks_str = f', constraints: ``{name} {sign} {number}``'
            else:
                checks_str = ''
            main_desc = f'-  ``{name}`` :raw-html:`<a id="{name}" title="Permalink to this parameter" href="#{name}">&#x1F517;&#xFE0E;</a>`, default = ``{default}``, type = {param_type}{options_str}{aliases_str}{checks_str}'
            params_to_write.append(main_desc)
            params_to_write.extend([f"{' ' * 3 * int(desc[0][-1])}-  {desc[1]}" for desc in param_desc['desc']])

    with open(params_rst) as original_params_file:
        all_lines = original_params_file.read()
        before, start_sep, _ = all_lines.partition('.. start params list\n\n')
        _, end_sep, after = all_lines.partition('\n\n.. end params list')

    with open(params_rst, "w") as new_params_file:
        new_params_file.write(before)
        new_params_file.write(start_sep)
        new_params_file.write('\n\n'.join(params_to_write))
        new_params_file.write(end_sep)
        new_params_file.write(after)


def gen_parameter_code(
    config_hpp: Path,
    config_out_cpp: Path
) -> Tuple[List[Tuple[str, int]], List[List[Dict[str, List]]]]:
    """Generate auto config file.

    Parameters
    ----------
    config_hpp : pathlib.Path
        Path to the config header file.
    config_out_cpp : pathlib.Path
        Path to the auto config file.

    Returns
    -------
    infos : tuple
        Tuple with names and content of sections.
    """
    keys, infos = get_parameter_infos(config_hpp)
    names = get_names(infos)
    alias = get_alias(infos)
    str_to_write = r"""/*!
 * Copyright (c) 2018 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \note
 * This file is auto generated by LightGBM\helpers\parameter_generator.py from LightGBM\include\LightGBM\config.h file.
 */
"""
    str_to_write += "#include<LightGBM/config.h>\nnamespace LightGBM {\n"
    # alias table
    str_to_write += "const std::unordered_map<std::string, std::string>& Config::alias_table() {\n"
    str_to_write += "  static std::unordered_map<std::string, std::string> aliases({\n"

    for pair in alias:
        str_to_write += f'  {{"{pair[0]}", "{pair[1]}"}},\n'
    str_to_write += "  });\n"
    str_to_write += "  return aliases;\n"
    str_to_write += "}\n\n"

    # names
    str_to_write += "const std::unordered_set<std::string>& Config::parameter_set() {\n"
    str_to_write += "  static std::unordered_set<std::string> params({\n"

    for name in names:
        str_to_write += f'  "{name}",\n'
    str_to_write += "  });\n"
    str_to_write += "  return params;\n"
    str_to_write += "}\n\n"
    # from strings
    str_to_write += "void Config::GetMembersFromString(const std::unordered_map<std::string, std::string>& params) {\n"
    str_to_write += '  std::string tmp_str = "";\n'
    for x in infos:
        for y in x:
            if "[doc-only]" in y:
                continue
            param_type = y["inner_type"][0]
            name = y["name"][0]
            checks = []
            if "check" in y:
                checks = y["check"]
            tmp = set_one_var_from_string(name, param_type, checks)
            str_to_write += tmp
    # tails
    str_to_write = f"{str_to_write.strip()}\n}}\n\n"
    str_to_write += "std::string Config::SaveMembersToString() const {\n"
    str_to_write += "  std::stringstream str_buf;\n"
    for x in infos:
        for y in x:
            if "[doc-only]" in y or "[no-save]" in y:
                continue
            param_type = y["inner_type"][0]
            name = y["name"][0]
            if "vector" in param_type:
                if "int8" in param_type:
                    str_to_write += f'  str_buf << "[{name}: " << Common::Join(Common::ArrayCast<int8_t, int>({name}), ",") << "]\\n";\n'
                else:
                    str_to_write += f'  str_buf << "[{name}: " << Common::Join({name}, ",") << "]\\n";\n'
            else:
                str_to_write += f'  str_buf << "[{name}: " << {name} << "]\\n";\n'
    # tails
    str_to_write += "  return str_buf.str();\n"
    str_to_write += "}\n\n"
    str_to_write += "}  // namespace LightGBM\n"
    with open(config_out_cpp, "w") as config_out_cpp_file:
        config_out_cpp_file.write(str_to_write)

    return keys, infos


if __name__ == "__main__":
    current_dir = Path(__file__).absolute().parent
    config_hpp = current_dir.parent / 'include' / 'LightGBM' / 'config.h'
    config_out_cpp = current_dir.parent / 'src' / 'io' / 'config_auto.cpp'
    params_rst = current_dir.parent / 'docs' / 'Parameters.rst'
    sections, descriptions = gen_parameter_code(config_hpp, config_out_cpp)
    gen_parameter_description(sections, descriptions, params_rst)
