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

#include "ResetInputByName.h"

#include <loco.h>

#include <luci/IR/CircleNodes.h>

#include <iostream>
#include <sstream>

namespace opselector
{

ResetInputByNamePass::ResetInputByNamePass(const std::string &args)
{
  std::stringstream ss(args);
  std::string new_input;

  while (std::getline(ss, new_input, ','))
  {
    _new_inputs.emplace_back(new_input);
  }
}

bool ResetInputByNamePass::run(luci::Module *module)
{
  for (uint32_t g = 0; g < module->size(); ++g)
  {
    auto graph = module->graph(g);

    for (auto node : loco::input_nodes(graph))
    {
      auto circle_input = loco::must_cast<luci::CircleInput *>(node);
      auto input_exclude = graph->nodes()->create<luci::CircleInputExclude>();
      input_exclude->dtype(loco::DataType::FLOAT32);
      input_exclude->from(circle_input);
    }

    for (auto node : loco::all_nodes(graph))
    {
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);

      if (find(_new_inputs.begin(), _new_inputs.end(), circle_node->name()) == _new_inputs.end())
        continue;

      for (uint32_t i = 0; i < circle_node->arity(); ++i)
      {
        auto circle_arg = loco::must_cast<luci::CircleNode *>(circle_node->arg(i));

        if (dynamic_cast<luci::CircleConst *>(circle_arg) != nullptr)
          continue;

        auto graph_input = graph->inputs()->create();
        graph_input->dtype(circle_arg->dtype());
        graph_input->name(circle_arg->name());
        {
          auto shape = std::make_unique<loco::TensorShape>();
          shape->rank(circle_arg->rank());
          for (uint32_t d = 0; d < circle_arg->rank(); ++d)
            shape->dim(d) = circle_arg->dim(d);
          graph_input->shape(std::move(shape));
        }

        auto circle_input = graph->nodes()->create<luci::CircleInput>();
        circle_input->index(graph_input->index());
        circle_input->dtype(circle_arg->dtype());
        circle_input->rank(circle_arg->rank());
        for (uint32_t d = 0; d < circle_arg->rank(); ++d)
          circle_input->dim(d) = circle_arg->dim(d);
        circle_input->name(circle_arg->name());

        loco::replace(circle_arg).with(circle_input);
      }
    }
  }

  return true;
}

} // namespace opselector
