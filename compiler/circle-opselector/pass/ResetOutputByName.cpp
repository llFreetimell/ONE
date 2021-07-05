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

#include "ResetOutputByName.h"

#include <loco.h>

#include <luci/IR/CircleNodes.h>

#include <iostream>
#include <sstream>

namespace opselector
{

ResetOutputByNamePass::ResetOutputByNamePass(const std::string &args)
{
  std::stringstream ss(args);
  std::string new_output;

  while (std::getline(ss, new_output, ','))
  {
    _new_outputs.emplace_back(new_output);
  }
}

bool ResetOutputByNamePass::run(luci::Module *module)
{
  for (uint32_t g = 0; g < module->size(); ++g)
  {
    auto graph = module->graph(g);

    for (auto node : loco::output_nodes(graph))
    {
      auto circle_output = loco::must_cast<luci::CircleOutput *>(node);
      auto output_exclude = graph->nodes()->create<luci::CircleOutputExclude>();
      output_exclude->dtype(loco::DataType::FLOAT32);
      circle_output->from(output_exclude);
    }

    for (auto node : loco::all_nodes(graph))
    {
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);

      if (find(_new_outputs.begin(), _new_outputs.end(), circle_node->name()) == _new_outputs.end())
        continue;

      auto graph_output = graph->outputs()->create();
      graph_output->dtype(circle_node->dtype());
      graph_output->name(circle_node->name());
      {
        auto shape = std::make_unique<loco::TensorShape>();
        shape->rank(circle_node->rank());
        for (uint32_t d = 0; d < circle_node->rank(); ++d)
          shape->dim(d) = circle_node->dim(d);
        graph_output->shape(std::move(shape));
      }

      auto circle_output = graph->nodes()->create<luci::CircleOutput>();
      circle_output->index(graph_output->index());
      circle_output->dtype(circle_node->dtype());
      circle_output->rank(circle_node->rank());
      for (uint32_t d = 0; d < circle_node->rank(); ++d)
        circle_output->dim(d) = circle_node->dim(d);
      circle_output->name(circle_node->name());
      circle_output->from(circle_node);
    }
  }

  return true;
}

} // namespace opselector
