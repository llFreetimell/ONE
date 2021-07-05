/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __CIRCLE_OPSELECTOR_RESET_INPUT_BY_NAME__
#define __CIRCLE_OPSELECTOR_RESET_INPUT_BY_NAME__

#include "SinglePass.h"

#include <string>
#include <vector>

namespace opselector
{

/**
 * @brief  Class to reset input by name
 *
 */
class ResetInputByNamePass final : public SinglePass
{
public:
  ResetInputByNamePass() = delete;

  ResetInputByNamePass(const std::string &args);

public:
  bool run(luci::Module *module) final;

private:
  std::vector<std::string> _new_inputs;
};

} // namespace opselector

#endif // __CIRCLE_OPSELECTOR_RESET_INPUT_BY_NAME__