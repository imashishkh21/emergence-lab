"""Tests for agent neural networks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestNetwork:
    """Tests for US-012: ActorCritic network."""
    
    def test_network_forward(self):
        """Test that network produces correct output shapes."""
        from src.agents.network import ActorCritic
        from src.configs import Config
        
        config = Config()
        
        # Create network
        network = ActorCritic(
            hidden_dims=config.agent.hidden_dims,
            num_actions=5,  # stay, up, down, left, right
        )
        
        # Initialize with dummy input
        key = jax.random.PRNGKey(42)
        obs_dim = 64  # Example observation dimension
        dummy_obs = jnp.zeros((obs_dim,))
        
        params = network.init(key, dummy_obs)
        
        # Forward pass
        logits, value, gate = network.apply(params, dummy_obs)

        # Check shapes
        assert logits.shape == (5,)  # 5 actions
        assert value.shape == ()  # scalar value
        assert gate.shape == (config.field.num_channels,)  # gate values
    
    def test_network_batched(self):
        """Test that network works with batched inputs."""
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        batch_size = 32
        
        dummy_obs = jnp.zeros((batch_size, obs_dim))
        params = network.init(key, dummy_obs[0])
        
        # Batched forward pass using vmap
        batched_apply = jax.vmap(network.apply, in_axes=(None, 0))
        logits, values, gates = batched_apply(params, dummy_obs)

        assert logits.shape == (32, 5)
        assert values.shape == (32,)
        assert gates.shape == (32, 4)  # default num_field_channels
    
    def test_network_initialization(self):
        """Test that initialization follows orthogonal scheme."""
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 32
        params = network.init(key, jnp.zeros((obs_dim,)))
        
        # Params should exist and be properly shaped
        assert params is not None
        
        # Check that params have reasonable magnitudes (orthogonal init)
        flat_params = jax.tree_util.tree_leaves(params)
        for p in flat_params:
            if p.ndim >= 2:  # Weight matrices
                assert jnp.abs(p).max() < 10.0  # Not exploding
    
    def test_network_different_configs(self):
        """Test network with various configurations."""
        from src.agents.network import ActorCritic
        
        key = jax.random.PRNGKey(42)
        obs_dim = 48
        
        configs = [
            (32, 32),
            (64, 64),
            (128, 128),
            (64, 64, 64),  # 3 layers
        ]
        
        for hidden_dims in configs:
            network = ActorCritic(hidden_dims=hidden_dims, num_actions=5)
            params = network.init(key, jnp.zeros((obs_dim,)))
            logits, value, gate = network.apply(params, jnp.zeros((obs_dim,)))

            assert logits.shape == (5,)
            assert value.shape == ()
            assert gate.shape == (4,)  # default num_field_channels


class TestActionSampling:
    """Tests for US-013: Action sampling from policy."""
    
    def test_action_sampling(self):
        """Test that action sampling works correctly."""
        from src.agents.policy import sample_actions
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        init_key, sample_key = jax.random.split(key)
        
        obs_dim = 64
        num_envs = 8
        num_agents = 4
        
        # Initialize params
        dummy_obs = jnp.zeros((obs_dim,))
        params = network.init(init_key, dummy_obs)
        
        # Batch of observations: (num_envs, num_agents, obs_dim)
        obs = jax.random.normal(sample_key, (num_envs, num_agents, obs_dim))
        
        actions, log_probs, values, entropy, gate = sample_actions(
            network, params, obs, sample_key
        )

        # Check shapes
        assert actions.shape == (num_envs, num_agents)
        assert log_probs.shape == (num_envs, num_agents)
        assert values.shape == (num_envs, num_agents)
        assert entropy.shape == (num_envs, num_agents)
        assert gate.shape == (num_envs, num_agents, 4)  # default num_field_channels
        
        # Actions should be valid (0-4)
        assert jnp.all(actions >= 0)
        assert jnp.all(actions < 5)
    
    def test_action_sampling_deterministic_eval(self):
        """Test deterministic action selection for evaluation."""
        from src.agents.policy import sample_actions, get_deterministic_actions
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        
        params = network.init(key, jnp.zeros((obs_dim,)))
        obs = jax.random.normal(key, (4, 2, obs_dim))  # 4 envs, 2 agents
        
        # Get deterministic actions
        actions, gate = get_deterministic_actions(network, params, obs)

        assert actions.shape == (4, 2)
        assert gate.shape == (4, 2, 4)  # default num_field_channels
        assert jnp.all(actions >= 0)
        assert jnp.all(actions < 5)
    
    def test_action_sampling_stochastic(self):
        """Test that stochastic sampling produces varied actions."""
        from src.agents.policy import sample_actions
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        
        params = network.init(key, jnp.zeros((obs_dim,)))
        obs = jax.random.normal(key, (100, 1, obs_dim))  # 100 samples
        
        # Sample multiple times with different keys
        all_actions = []
        for i in range(10):
            sample_key = jax.random.PRNGKey(i)
            actions, _, _, _, _gate = sample_actions(network, params, obs, sample_key)
            all_actions.append(actions)
        
        all_actions = jnp.stack(all_actions)
        
        # Should have variety in actions (not all same)
        assert len(jnp.unique(all_actions)) > 1
    
    def test_action_sampling_jit_compatible(self):
        """Test that action sampling works with JIT."""
        from src.agents.policy import sample_actions
        from src.agents.network import ActorCritic
        
        network = ActorCritic(hidden_dims=(64, 64), num_actions=5)
        
        key = jax.random.PRNGKey(42)
        obs_dim = 64
        
        params = network.init(key, jnp.zeros((obs_dim,)))
        
        @jax.jit
        def jit_sample(obs, key):
            return sample_actions(network, params, obs, key)

        obs = jax.random.normal(key, (4, 2, obs_dim))
        actions, log_probs, values, entropy, gate = jit_sample(obs, key)

        assert actions.shape == (4, 2)
        assert gate.shape == (4, 2, 4)  # default num_field_channels


class TestAgentSpecificHeads:
    """Tests for Phase 4 US-001: Agent-Specific Policy Heads."""

    def test_agent_specific_heads(self):
        """Test that AgentSpecificActorCritic produces correct output shapes."""
        from src.agents.network import AgentSpecificActorCritic

        n_agents = 4
        network = AgentSpecificActorCritic(
            hidden_dims=(64, 64), num_actions=5, n_agents=n_agents
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 64
        dummy_obs = jnp.zeros((obs_dim,))
        agent_id = jnp.int32(0)

        params = network.init(key, dummy_obs, agent_id)

        # Forward pass with agent_id (AgentSpecificActorCritic returns 2 values)
        logits, value = network.apply(params, dummy_obs, agent_id)

        assert logits.shape == (5,), f"Expected (5,), got {logits.shape}"
        assert value.shape == (), f"Expected scalar, got {value.shape}"

    def test_different_agents_different_outputs(self):
        """Test that different agent_ids produce different outputs."""
        from src.agents.network import AgentSpecificActorCritic

        n_agents = 4
        network = AgentSpecificActorCritic(
            hidden_dims=(32, 32), num_actions=5, n_agents=n_agents
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.ones((obs_dim,))  # Non-zero input for differentiation
        params = network.init(key, dummy_obs, jnp.int32(0))

        # Get outputs for each agent (AgentSpecificActorCritic returns 2 values)
        outputs = []
        for i in range(n_agents):
            logits, value = network.apply(params, dummy_obs, jnp.int32(i))
            outputs.append((np.array(logits), float(value)))

        # At least some agents should produce different outputs
        # (different heads have different random initializations)
        logits_arrays = [o[0] for o in outputs]
        any_different = False
        for i in range(len(logits_arrays)):
            for j in range(i + 1, len(logits_arrays)):
                if not np.allclose(logits_arrays[i], logits_arrays[j], atol=1e-6):
                    any_different = True
                    break
        assert any_different, "All agent heads produced identical outputs"

    def test_shared_encoder_weights(self):
        """Test that encoder weights are shared across agents."""
        from src.agents.network import AgentSpecificActorCritic

        n_agents = 4
        network = AgentSpecificActorCritic(
            hidden_dims=(32, 32), num_actions=5, n_agents=n_agents
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        params = network.init(key, jnp.zeros((obs_dim,)), jnp.int32(0))

        # Verify param structure via top-level keys
        param_dict = params["params"]
        param_names = list(param_dict.keys())

        # There should be encoder Dense layers (shared)
        encoder_names = [n for n in param_names if n.startswith("Dense_")]
        assert len(encoder_names) >= 2, "Expected at least 2 shared encoder layers"

        # There should be per-agent head params (actor_head_N and critic_head_N)
        actor_head_names = [n for n in param_names if n.startswith("actor_head_")]
        critic_head_names = [n for n in param_names if n.startswith("critic_head_")]
        assert len(actor_head_names) == n_agents, (
            f"Expected {n_agents} actor heads, found {len(actor_head_names)}"
        )
        assert len(critic_head_names) == n_agents, (
            f"Expected {n_agents} critic heads, found {len(critic_head_names)}"
        )

    def test_agent_id_none_uses_head_0(self):
        """Test backward compatibility: agent_id=None uses head 0."""
        from src.agents.network import AgentSpecificActorCritic

        n_agents = 4
        network = AgentSpecificActorCritic(
            hidden_dims=(32, 32), num_actions=5, n_agents=n_agents
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.ones((obs_dim,))
        params = network.init(key, dummy_obs, jnp.int32(0))

        # Call without agent_id (AgentSpecificActorCritic returns 2 values)
        logits_none, value_none = network.apply(params, dummy_obs)
        # Call with agent_id=0
        logits_0, value_0 = network.apply(params, dummy_obs, jnp.int32(0))

        np.testing.assert_allclose(
            np.array(logits_none), np.array(logits_0), atol=1e-6
        )
        np.testing.assert_allclose(float(value_none), float(value_0), atol=1e-6)

    def test_agent_id_out_of_range_clamped(self):
        """Test that out-of-range agent_ids are safely clamped."""
        from src.agents.network import AgentSpecificActorCritic

        n_agents = 4
        network = AgentSpecificActorCritic(
            hidden_dims=(32, 32), num_actions=5, n_agents=n_agents
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.zeros((obs_dim,))
        params = network.init(key, dummy_obs, jnp.int32(0))

        # agent_id beyond n_agents should be clamped to last head
        # (AgentSpecificActorCritic returns 2 values)
        logits_big, value_big = network.apply(
            params, dummy_obs, jnp.int32(100)
        )
        logits_last, value_last = network.apply(
            params, dummy_obs, jnp.int32(n_agents - 1)
        )

        np.testing.assert_allclose(
            np.array(logits_big), np.array(logits_last), atol=1e-6
        )

    def test_jit_compatible(self):
        """Test that agent-specific heads work with JIT compilation."""
        from src.agents.network import AgentSpecificActorCritic

        n_agents = 8
        network = AgentSpecificActorCritic(
            hidden_dims=(32, 32), num_actions=5, n_agents=n_agents
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.zeros((obs_dim,))
        params = network.init(key, dummy_obs, jnp.int32(0))

        @jax.jit
        def forward(obs, agent_id):
            return network.apply(params, obs, agent_id)

        obs = jax.random.normal(key, (obs_dim,))
        for i in range(n_agents):
            logits, value = forward(obs, jnp.int32(i))
            assert logits.shape == (5,)
            assert value.shape == ()

    def test_vmap_over_agents(self):
        """Test that we can vmap the forward pass over agent_ids."""
        from src.agents.network import AgentSpecificActorCritic

        n_agents = 4
        network = AgentSpecificActorCritic(
            hidden_dims=(32, 32), num_actions=5, n_agents=n_agents
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.zeros((obs_dim,))
        params = network.init(key, dummy_obs, jnp.int32(0))

        # Create batch of observations and agent_ids
        batch_obs = jax.random.normal(key, (n_agents, obs_dim))
        agent_ids = jnp.arange(n_agents, dtype=jnp.int32)

        # vmap over both obs and agent_id
        batched_apply = jax.vmap(
            lambda obs, aid: network.apply(params, obs, aid),
            in_axes=(0, 0),
        )
        logits, values = batched_apply(batch_obs, agent_ids)

        assert logits.shape == (n_agents, 5)
        assert values.shape == (n_agents,)

    def test_gradients_flow_through_correct_head(self):
        """Test that gradients only flow through the selected agent's head."""
        from src.agents.network import AgentSpecificActorCritic

        n_agents = 4
        network = AgentSpecificActorCritic(
            hidden_dims=(32, 32), num_actions=5, n_agents=n_agents
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.ones((obs_dim,))
        params = network.init(key, dummy_obs, jnp.int32(0))

        # Compute gradient w.r.t. params for agent 0
        def loss_fn(p, agent_id):
            logits, value = network.apply(p, dummy_obs, agent_id)
            return jnp.sum(logits) + value

        grads_agent0 = jax.grad(loss_fn)(params, jnp.int32(0))
        grads_agent1 = jax.grad(loss_fn)(params, jnp.int32(1))

        # actor_head_0 should have non-zero grads for agent 0, zero for agent 1
        head_0_actor_grad_for_agent0 = jax.tree_util.tree_leaves(
            grads_agent0["params"]["actor_head_0"]
        )
        head_0_actor_grad_for_agent1 = jax.tree_util.tree_leaves(
            grads_agent1["params"]["actor_head_0"]
        )

        # Agent 0's gradient should have non-zero values in actor_head_0
        has_nonzero_0 = any(
            float(jnp.abs(g).sum()) > 1e-8 for g in head_0_actor_grad_for_agent0
        )
        assert has_nonzero_0, "Agent 0 should have gradients in actor_head_0"

        # Agent 1's gradient should have zero values in actor_head_0
        all_zero_1 = all(
            float(jnp.abs(g).sum()) < 1e-8 for g in head_0_actor_grad_for_agent1
        )
        assert all_zero_1, "Agent 1 should NOT have gradients in actor_head_0"

        # Shared encoder should have non-zero grads for both agents
        encoder_grad_0 = jax.tree_util.tree_leaves(
            grads_agent0["params"]["Dense_0"]
        )
        encoder_grad_1 = jax.tree_util.tree_leaves(
            grads_agent1["params"]["Dense_0"]
        )
        has_encoder_grad_0 = any(
            float(jnp.abs(g).sum()) > 1e-8 for g in encoder_grad_0
        )
        has_encoder_grad_1 = any(
            float(jnp.abs(g).sum()) > 1e-8 for g in encoder_grad_1
        )
        assert has_encoder_grad_0, "Agent 0 should have encoder gradients"
        assert has_encoder_grad_1, "Agent 1 should have encoder gradients"

    def test_config_agent_architecture(self):
        """Test that config has agent_architecture field."""
        from src.configs import Config

        config = Config()
        assert config.agent.agent_architecture == "shared"

        # Test agent_heads option
        config.agent.agent_architecture = "agent_heads"
        assert config.agent.agent_architecture == "agent_heads"

    def test_different_hidden_dims(self):
        """Test AgentSpecificActorCritic with various hidden dim configurations."""
        from src.agents.network import AgentSpecificActorCritic

        key = jax.random.PRNGKey(42)
        obs_dim = 48
        n_agents = 4

        configs = [
            (32, 32),
            (64, 64),
            (64, 64, 64),  # 3 layers
        ]

        for hidden_dims in configs:
            network = AgentSpecificActorCritic(
                hidden_dims=hidden_dims, num_actions=5, n_agents=n_agents
            )
            params = network.init(key, jnp.zeros((obs_dim,)), jnp.int32(0))

            for agent_id in range(n_agents):
                logits, value = network.apply(
                    params, jnp.zeros((obs_dim,)), jnp.int32(agent_id)
                )
                assert logits.shape == (5,)
                assert value.shape == ()


class TestAgentEmbedding:
    """Tests for Phase 4 US-002: Agent ID Embedding."""

    def test_agent_embedding(self):
        """Test that ActorCritic with embedding produces correct output shapes."""
        from src.agents.network import ActorCritic

        n_agents = 8
        embed_dim = 8
        network = ActorCritic(
            hidden_dims=(64, 64),
            num_actions=5,
            agent_embed_dim=embed_dim,
            n_agents=n_agents,
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 64
        dummy_obs = jnp.zeros((obs_dim,))
        agent_id = jnp.int32(0)

        params = network.init(key, dummy_obs, agent_id)
        logits, value, gate = network.apply(params, dummy_obs, agent_id)

        assert logits.shape == (5,)
        assert value.shape == ()

    def test_embedding_creates_param(self):
        """Test that embedding creates agent_embedding parameter in param tree."""
        from src.agents.network import ActorCritic

        n_agents = 4
        embed_dim = 8
        network = ActorCritic(
            hidden_dims=(32, 32),
            num_actions=5,
            agent_embed_dim=embed_dim,
            n_agents=n_agents,
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        params = network.init(key, jnp.zeros((obs_dim,)), jnp.int32(0))

        param_dict = params["params"]
        assert "agent_embedding" in param_dict, (
            f"Expected 'agent_embedding' in params, got {list(param_dict.keys())}"
        )
        # Embedding table should be (n_agents, embed_dim)
        embedding_table = param_dict["agent_embedding"]["embedding"]
        assert embedding_table.shape == (n_agents, embed_dim)

    def test_no_embedding_when_disabled(self):
        """Test that no embedding param exists when agent_embed_dim=0."""
        from src.agents.network import ActorCritic

        network = ActorCritic(
            hidden_dims=(32, 32), num_actions=5, agent_embed_dim=0
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        params = network.init(key, jnp.zeros((obs_dim,)))

        param_dict = params["params"]
        assert "agent_embedding" not in param_dict, (
            "Should not have agent_embedding when embed_dim=0"
        )

    def test_different_agents_different_embeddings(self):
        """Test that different agent_ids produce different outputs via embedding."""
        from src.agents.network import ActorCritic

        n_agents = 4
        embed_dim = 8
        network = ActorCritic(
            hidden_dims=(32, 32),
            num_actions=5,
            agent_embed_dim=embed_dim,
            n_agents=n_agents,
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.ones((obs_dim,))
        params = network.init(key, dummy_obs, jnp.int32(0))

        # With embedding, same observation + different agent_id should give different outputs
        outputs = []
        for i in range(n_agents):
            logits, value, gate = network.apply(params, dummy_obs, jnp.int32(i))
            outputs.append(np.array(logits))

        # At least some agents should differ (different embeddings → different encoder input)
        any_different = False
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if not np.allclose(outputs[i], outputs[j], atol=1e-6):
                    any_different = True
                    break
        assert any_different, "Different agent_ids should produce different outputs via embedding"

    def test_backward_compat_no_embedding(self):
        """Test that ActorCritic with embed_dim=0 works without agent_id."""
        from src.agents.network import ActorCritic

        network = ActorCritic(
            hidden_dims=(32, 32),
            num_actions=5,
            agent_embed_dim=0,
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.zeros((obs_dim,))
        # Init without agent_id — no embedding
        params = network.init(key, dummy_obs)

        # Call without agent_id — should work fine
        logits, value, gate = network.apply(params, dummy_obs)
        assert logits.shape == (5,)
        assert value.shape == ()

        # Also works with agent_id=None explicitly
        logits2, value2, gate2 = network.apply(params, dummy_obs, None)
        np.testing.assert_allclose(np.array(logits), np.array(logits2), atol=1e-6)

    def test_agent_specific_with_embedding(self):
        """Test AgentSpecificActorCritic with embedding enabled."""
        from src.agents.network import AgentSpecificActorCritic

        n_agents = 4
        embed_dim = 8
        network = AgentSpecificActorCritic(
            hidden_dims=(32, 32),
            num_actions=5,
            n_agents=n_agents,
            agent_embed_dim=embed_dim,
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.ones((obs_dim,))
        agent_id = jnp.int32(0)

        params = network.init(key, dummy_obs, agent_id)

        # Check embedding exists
        assert "agent_embedding" in params["params"]

        # Forward pass for each agent (AgentSpecificActorCritic returns 2 values)
        for i in range(n_agents):
            logits, value = network.apply(params, dummy_obs, jnp.int32(i))
            assert logits.shape == (5,)
            assert value.shape == ()

    def test_embedding_jit_compatible(self):
        """Test that embedding works with JIT compilation."""
        from src.agents.network import ActorCritic

        n_agents = 8
        embed_dim = 8
        network = ActorCritic(
            hidden_dims=(32, 32),
            num_actions=5,
            agent_embed_dim=embed_dim,
            n_agents=n_agents,
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.zeros((obs_dim,))
        params = network.init(key, dummy_obs, jnp.int32(0))

        @jax.jit
        def forward(obs, agent_id):
            return network.apply(params, obs, agent_id)

        obs = jax.random.normal(key, (obs_dim,))
        for i in range(n_agents):
            logits, value, gate = forward(obs, jnp.int32(i))
            assert logits.shape == (5,)
            assert value.shape == ()

    def test_embedding_vmap_over_agents(self):
        """Test that embedding works with vmap over agent_ids."""
        from src.agents.network import ActorCritic

        n_agents = 4
        embed_dim = 8
        network = ActorCritic(
            hidden_dims=(32, 32),
            num_actions=5,
            agent_embed_dim=embed_dim,
            n_agents=n_agents,
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        params = network.init(key, jnp.zeros((obs_dim,)), jnp.int32(0))

        batch_obs = jax.random.normal(key, (n_agents, obs_dim))
        agent_ids = jnp.arange(n_agents, dtype=jnp.int32)

        batched_apply = jax.vmap(
            lambda obs, aid: network.apply(params, obs, aid),
            in_axes=(0, 0),
        )
        logits, values, gates = batched_apply(batch_obs, agent_ids)

        assert logits.shape == (n_agents, 5)
        assert values.shape == (n_agents,)

    def test_embedding_gradients_flow(self):
        """Test that gradients flow through the embedding table."""
        from src.agents.network import ActorCritic

        n_agents = 4
        embed_dim = 8
        network = ActorCritic(
            hidden_dims=(32, 32),
            num_actions=5,
            agent_embed_dim=embed_dim,
            n_agents=n_agents,
        )

        key = jax.random.PRNGKey(42)
        obs_dim = 32
        dummy_obs = jnp.ones((obs_dim,))
        params = network.init(key, dummy_obs, jnp.int32(0))

        def loss_fn(p, agent_id):
            logits, value, gate = network.apply(p, dummy_obs, agent_id)
            return jnp.sum(logits) + value

        grads = jax.grad(loss_fn)(params, jnp.int32(1))

        # Embedding should have gradients
        embed_grad = grads["params"]["agent_embedding"]["embedding"]
        assert embed_grad.shape == (n_agents, embed_dim)

        # Only the row for agent 1 should have non-zero gradients
        assert float(jnp.abs(embed_grad[1]).sum()) > 1e-8, (
            "Agent 1 embedding should have gradients"
        )
        assert float(jnp.abs(embed_grad[0]).sum()) < 1e-8, (
            "Agent 0 embedding should NOT have gradients when agent_id=1"
        )

    def test_config_agent_embed_dim(self):
        """Test that config has agent_embed_dim field with default 0."""
        from src.configs import Config

        config = Config()
        assert config.agent.agent_embed_dim == 0

        config.agent.agent_embed_dim = 8
        assert config.agent.agent_embed_dim == 8
