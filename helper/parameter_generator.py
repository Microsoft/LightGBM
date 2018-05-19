def GetParameterInfos(config_hpp="..\\include\\LightGBM\\config.h"):
    config_hpp_file = open("..\\include\\LightGBM\\config.h")
    is_inparameter = False
    parameter_group = None
    cur_key = None
    cur_info = {}
    keys = []
    member_infos = []
    for line in config_hpp_file.readlines():
        if "#pragma region Parameters" in line:
            is_inparameter = True
        elif "#pragma region" in line and "Parameters" in line:
            cur_key = line.split("region")[1].strip()
            keys.append(cur_key)
            member_infos.append([])
        elif '#pragma endregion' in line:
            if cur_key is not None:
                cur_key = None
            elif is_inparameter:
                is_inparameter = False
        elif cur_key is not None:
            line = line.strip()
            if line.startswith("//"):
                tokens = line.split("//")[1].split("=")
                key = tokens[0].strip()
                val = '='.join(tokens[1:]).strip()
                if key not in cur_info:
                    if key == "descl2":
                        cur_info["desc"] = []
                    else:
                        cur_info[key] = []
                if key == "desc":
                    cur_info["desc"].append(["l1", val])
                elif key == "descl2":
                    cur_info["desc"].append(["l2", val])
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
    config_hpp_file.close()
    return (keys, member_infos)


def GetNames(infos):
    names = []
    for x in infos:
        for y in x:
            names.append(y["name"][0])
    return names


def GetAlias(infos):
    pairs = []
    for x in infos:
        for y in x:
            if "alias" in y:
                name = y["name"][0]
                alias = y["alias"][0].split(',')
                for name2 in alias:
                    pairs.append([name2.strip(), name])
    return pairs


def SetOneVarFromString(name, type, checks):
    ret = ""
    univar_mapper = {"int": "GetInt", "double": "GetDouble", "bool": "GetBool", "std::string": "GetString"}
    if "vector" not in type:
        ret += "  %s(params, \"%s\", &%s);\n" % (univar_mapper[type], name, name)
        if len(checks) > 0:
            for check in checks:
                ret += "  CHECK(%s %s);\n" % (name, check)
        ret += "\n"
    else:
        ret += "  if (GetString(params, \"%s\", &tmp_str)) {\n" % (name)
        type2 = type.split("<")[1][:-1]
        if type2 == "std::string":
            ret += "    %s = Common::Split(tmp_str.c_str(), ',');\n" % (name)
        else:
            ret += "    %s = Common::StringToArray<%s>(tmp_str, ',');\n" % (name, type2)
        ret += "  }\n\n"
    return ret


def GenParameterCode(config_hpp="..\\include\\LightGBM\\config.h", config_out_cpp="..\\src\\io\\config_auto.cpp"):
    keys, infos = GetParameterInfos(config_hpp)
    names = GetNames(infos)
    alias = GetAlias(infos)
    str_to_write = "#include<LightGBM/config.h>\nnamespace LightGBM {\n"
    # alias table
    str_to_write += "std::unordered_map<std::string, std::string> Config::alias_table({\n"
    for pair in alias:
        str_to_write += "  {\"%s\", \"%s\"}, \n" % (pair[0], pair[1])
    str_to_write += "});\n\n"
    # names
    str_to_write += "std::unordered_set<std::string> Config::parameter_set({\n"
    for name in names:
        str_to_write += "  \"%s\", \n" % (name)
    str_to_write += "});\n\n"
    # from strings
    str_to_write += "void Config::GetMembersFromString(const std::unordered_map<std::string, std::string>& params) {\n"
    str_to_write += "  std::string tmp_str = \"\";\n"
    for x in infos:
        for y in x:
            if "[doc-only]" in y:
                continue
            type = y["inner_type"][0]
            name = y["name"][0]
            checks = []
            if "check" in y:
                checks = y["check"]
            tmp = SetOneVarFromString(name, type, checks)
            str_to_write += tmp
    # tails
    str_to_write += "}\n\n"
    str_to_write += "std::string Config::SaveMembersToString() const {\n"
    str_to_write += "  std::stringstream str_buf;\n"
    for x in infos:
        for y in x:
            if "[doc-only]" in y:
                continue
            type = y["inner_type"][0]
            name = y["name"][0]
            if "vector" in type:
                str_to_write += "  str_buf << \"%s=\" << Common::Join(%s,\",\") << \"\\n\";\n" % (name, name)
            else:
                str_to_write += "  str_buf << \"%s=\" << %s << \"\\n\";\n" % (name, name)
    # tails
    str_to_write += "  return str_buf.str();\n"
    str_to_write += "}\n\n"
    str_to_write += "}\n"
    with open(config_out_cpp, "w") as config_out_cpp_file:
        config_out_cpp_file.write(str_to_write)


GenParameterCode()
