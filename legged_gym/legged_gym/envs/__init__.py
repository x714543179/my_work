# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.go2w.go2w_dreamwaq.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO
from .go2w.go2w_dreamwaq.go2w_robot import Go2w
from legged_gym.envs.go2w.go2w_sf.go2w_sf_config import (
    GO2WSfCfg,
    GO2WSfCfgPPO,
    GO2WSfTrotCfg,
    GO2WSfTrotCfgPPO,
)
from .go2w.go2w_sf.go2w_sf_robot import Go2wSf
from legged_gym.envs.go1.go1_omni_jump.go1_omni_jump_config import (
    Go1OmniJumpCfg,
    Go1OmniJumpCfgPPO,
)
from .go1.go1_omni_jump.go1_omni_jump_robot import Go1OmniJump
from legged_gym.envs.nezha.nezha_jump import (
    NezhaJump,
    NezhaJumpCfg,
    NezhaJumpCfgPPO,
)

# from .nezha.nezha_config import NEZHARoughCfg, NEZHARoughCfgPPO
# from .nezha.nezha_robot import NEZHA
from legged_gym.utils.task_registry import task_registry


task_registry.register( "go2w", Go2w, GO2WRoughCfg(), GO2WRoughCfgPPO())

task_registry.register("go2w_sf_trot", Go2wSf, GO2WSfTrotCfg(), GO2WSfTrotCfgPPO())
task_registry.register("go2w_sf", Go2wSf, GO2WSfCfg(), GO2WSfCfgPPO())
task_registry.register("go1_omni_jump", Go1OmniJump, Go1OmniJumpCfg(), Go1OmniJumpCfgPPO())
task_registry.register("nezha_jump", NezhaJump, NezhaJumpCfg(), NezhaJumpCfgPPO())

# task_registry.register("nezha", NEZHA, NEZHARoughCfg(), NEZHARoughCfgPPO() )
