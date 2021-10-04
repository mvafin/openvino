// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ngraph {
namespace frontend {
namespace common {
#ifdef _WIN32
const char PATH_SEPARATOR = '\\';
#    if defined(ENABLE_UNICODE_PATH_SUPPORT)
const wchar_t WPATH_SEPARATOR = L'\\';
#    endif
#else
const char PATH_SEPARATOR = '/';
#endif

template <typename T>
inline std::basic_string<T> get_path_sep() {
    return std::basic_string<T>{PATH_SEPARATOR};
}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
inline std::basic_string<wchar_t> get_path_sep() {
    return std::basic_string<wchar_t>{WPATH_SEPARATOR};
}
#endif

template <typename T>
bool ends_with(const std::basic_string<T>& str, const std::basic_string<T>& suffix) {
    if (str.length() >= suffix.length()) {
        return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
    }
    return false;
}

inline std::string str_join(const std::vector<std::string>& strs, const char* sep) {
    std::ostringstream str;
    std::copy(strs.begin(), strs.end(), std::ostream_iterator<std::string>(str, sep));
    return str.str();
}

}  // namespace common
}  // namespace frontend
}  // namespace ngraph
