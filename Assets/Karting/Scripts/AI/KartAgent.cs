using MLAgents;
using KartGame.KartSystems;
using UnityEngine;

namespace KartGame.AI
{
    /// <summary>
    /// Sensors hold information such as the position of rotation of the origin of the raycast and its hit threshold
    /// to consider a "crash".
    /// </summary>
    [System.Serializable]
    public struct Sensor
    {
        public Transform Transform;
        public float HitThreshold;
    }

    /// <summary>
    /// We only want certain behaviours when the agent runs.
    /// Training would allow certain functions such as OnAgentReset() be called and execute, while Inferencing will
    /// assume that the agent will continuously run and not reset.
    /// </summary>
    public enum AgentMode
    {
        Training,
        Inferencing
    }

    /// <summary>
    /// The KartAgent will drive the inputs for the KartController.
    /// </summary>
    public class KartAgent : Agent, IInput
    {
        /// <summary>
        /// How many actions are we going to support when we use our own custom heuristic? Right now we want the X/Y
        /// axis for acceleration and steering.
        /// </summary>
        const int LocalActionSize = 2;

        #region Training Modes
        [Tooltip("Are we training the agent or is the agent production ready?")]
        public AgentMode Mode = AgentMode.Training;
        [Tooltip("What is the initial checkpoint the agent will go to? This value is only for inferencing.")]
        public ushort InitCheckpointIndex;
        #endregion

        #region Senses
        [Header("Observation Params")]
        [Tooltip("How far should the agent shoot raycasts to detect the world?")]
        public float RaycastDistance;
        [Tooltip("What objects should the raycasts hit and detect?")]
        public LayerMask Mask;
        [Tooltip("Sensors contain ray information to sense out the world, you can have as many sensors as you need.")]
        public Sensor[] Sensors;

        [Header("Checkpoints")]
        [Tooltip("What are the series of checkpoints for the agent to seek and pass through?")]
        public Collider[] Colliders;
        [Tooltip("What layer are the checkpoints on? This should be an exclusive layer for the agent to use.")]
        public LayerMask CheckpointMask;

        [Space]
        [Tooltip("Would the agent need a custom transform to be able to raycast and hit the track? " +
            "If not assigned, then the root transform will be used.")]
        public Transform AgentSensorTransform;
        #endregion

        #region Rewards
        [Header("Rewards")]
        [Tooltip("What penatly is given when the agent crashes?")]
        public float HitPenalty = -1f;
        [Tooltip("How much reward is given when the agent successfully passes the checkpoints?")]
        public float PassCheckpointReward;
        [Tooltip("Should typically be a small value, but we reward the agent for moving in the right direction.")]
        public float TowardsCheckpointReward;
        [Tooltip("Typically if the agent moves faster, we want to reward it for finishing the track quickly.")]
        public float SpeedReward;
        [Tooltip("When the track is completed faster than the previous round, we want to give a reward.")]
        public float _fasterTrackReward = 0.5f;
        [Tooltip("Punishment when the agent is not moving at all.")]
        public float _idlePunishment = -0.5f;
        [Tooltip("When distance to next point is smaller than last check, give reward.")]
        public float _movedTowardsPointReward = 0.001f;
        #endregion

        #region ResetParams
        [Header("Inference Reset Params")]
        [Tooltip("What is the unique mask that the agent should detect when it falls out of the track?")]
        public LayerMask OutOfBoundsMask;
        [Tooltip("What are the layers we want to detect for the track and the ground?")]
        public LayerMask TrackMask;
        [Tooltip("How far should the ray be when casted? For larger karts - this value should be larger too.")]
        public float GroundCastDistance;

        /* Starting at model_04: no more random position selection for training */
        [Tooltip("The position where agents start when they reset.")]
        public Transform _startPosition;
        #endregion

        #region Debugging
        [Header("Debug Option")]
        [Tooltip("Should we visualize the rays that the agent draws?")]
        public bool ShowRaycasts;
        #endregion

        ArcadeKart kart;
        float acceleration;
        float steering;
        float[] localActions;
        int checkpointIndex;

        /* Agent reward model_05 */
        float _wallHitTimer = 0.0f;
        bool _wallHit = false;
        int _indexHitSensor = 0;
        public float _resetTimeAfterHit = 3.0f;

        float _idleTimer = 0.0f;
        bool _isIdle = false;
        public float _resetIdleTimerAfter = 5.0f;
        Vector3 _lastPosition = Vector3.zero;
        float _newPositionTimer = 0.0f;
        float _newPositionAfter = 1.5f;

        /* Agent reward model_04 */
        float _timer = 0.0f;
        float _previousRoundTime = 0.0f;
        bool _roundStarted = false;

        void Awake()
        {
            kart = GetComponent<ArcadeKart>();
            if (AgentSensorTransform == null)
            {
                AgentSensorTransform = transform;
            }
        }

        void Start()
        {
            localActions = new float[LocalActionSize];

            // If the agent is training, then at the start of the simulation, pick a random checkpoint to train the agent. 
            /* UPDATE: no more random selection of points, starting at model_04*/
            AgentReset();

            if (Mode == AgentMode.Inferencing)
            {
                checkpointIndex = InitCheckpointIndex;
            }
        }

        void LateUpdate()
        {
            /* Reward model_05 
            if (_roundStarted)
                _timer += Time.deltaTime;
             */

            /* Reward model_06 */
            if (_wallHit)
                _wallHitTimer += Time.deltaTime;

            /* Reward model_07 */
            if (_isIdle)
                _idleTimer += Time.deltaTime;

            _newPositionTimer += Time.deltaTime;

            switch (Mode)
            {
                case AgentMode.Inferencing:
                    if (ShowRaycasts)
                    {
                        Debug.DrawRay(transform.position, Vector3.down * GroundCastDistance, Color.cyan);
                    }
                    // We want to place the agent back on the track if the agent happens to launch itself outside of the track.
                    if (Physics.Raycast(transform.position, Vector3.down, out var hit, GroundCastDistance, TrackMask)
                        && ((1 << hit.collider.gameObject.layer) & OutOfBoundsMask) > 0)
                    {
                        // Reset the agent back to its last known agent checkpoint
                        Transform checkpoint = Colliders[checkpointIndex].transform;
                        transform.localRotation = checkpoint.rotation;
                        transform.position = checkpoint.position;
                        kart.Rigidbody.velocity = default;
                        acceleration = steering = 0f;
                    }
                    break;
            }
        }

        void OnTriggerEnter(Collider other)
        {
            int maskedValue = 1 << other.gameObject.layer;
            int triggered = maskedValue & CheckpointMask;

            FindCheckpointIndex(other, out int index);

            // Ensure that the agent touched the checkpoint and the new index is greater than the m_CheckpointIndex.
            if (triggered > 0 && index > checkpointIndex || index == 0 && checkpointIndex == Colliders.Length - 1)
            {
                AddReward(PassCheckpointReward);
                checkpointIndex = index;
            }

            /* Rewarding model 4/5, if agent hits the start: timer should run, if agents hits the start again then the round is over and the time is saved. 
            if (other.gameObject.layer == 15)
            {
                if (_timer > 1.0f)
                {
                    if (_roundStarted)
                    {
                        if (_previousRoundTime < _timer)
                            AddReward(_fasterTrackReward);
                        _previousRoundTime = _timer;
                    }
                    _timer = 0.0f;
                }
                else
                {
                    _roundStarted = true;
                }
            }
             */
        }

        void FindCheckpointIndex(Collider checkPoint, out int index)
        {
            for (int i = 0; i < Colliders.Length; i++)
            {
                if (Colliders[i].GetInstanceID() == checkPoint.GetInstanceID())
                {
                    index = i;
                    return;
                }
            }
            index = -1;
        }

        float Sign(float value)
        {
            if (value > 0)
            {
                return 1;
            }
            else if (value < 0)
            {
                return -1;
            }
            return 0;
        }

        void InterpretDiscreteActions(float[] actions)
        {
            steering = actions[0] - 1f;
            acceleration = Mathf.FloorToInt(actions[1]) == 1 ? 1 : 0;
        }

        public override void CollectObservations()
        {
            //AddVectorObs(kart.LocalSpeed());

            /* Model_05: extra observations 
              AddVectorObs(_previousRoundTime); // float = 1
              AddVectorObs(_timer); //float = 1
             */

            // Add an observation for direction of the agent to the next checkpoint.
            var next = (checkpointIndex + 1) % Colliders.Length;
            var nextCollider = Colliders[next];
            var direction = (nextCollider.transform.position - kart.transform.position).normalized;
            AddVectorObs(Vector3.Dot(kart.Rigidbody.velocity.normalized, direction));

            if (ShowRaycasts)
            {
                Debug.DrawLine(AgentSensorTransform.position, nextCollider.transform.position, Color.magenta);
            }

            for (int i = 0; i < Sensors.Length; i++)
            {
                var current = Sensors[i];
                var xform = current.Transform;
                var hit = Physics.Raycast(AgentSensorTransform.position, xform.forward, out var hitInfo,
                    RaycastDistance, Mask, QueryTriggerInteraction.Ignore);

                if (ShowRaycasts)
                {
                    Debug.DrawRay(AgentSensorTransform.position, xform.forward * RaycastDistance, Color.green);
                    Debug.DrawRay(AgentSensorTransform.position, xform.forward * RaycastDistance * current.HitThreshold,
                        Color.red);
                }

                var hitDistance = (hit ? hitInfo.distance : RaycastDistance) / RaycastDistance;
                AddVectorObs(hitDistance);

                /* Model_06 */
                if (hitDistance < current.HitThreshold)
                {
                    /* Model_06 uses timer to look how long the wall was hit. */
                    if (_wallHit == false)
                    {
                        _indexHitSensor = i;
                        _wallHit = true;
                    }


                    //Only reset the agent after hitting the wall for _resetTimeAfterHit in seconds
                    if (_wallHitTimer > _resetTimeAfterHit)
                    {
                        //Normal reward
                        AddReward(HitPenalty);
                        //Normal reward
                        Done();
                        //Normal reward
                        AgentReset();
                    }
                }
                else if (_indexHitSensor == i && _wallHitTimer > 0.1f) // Model_06
                {
                    _wallHit = false;
                    _wallHitTimer = 0.0f;
                }
            }
            /* Model_06 */
            AddVectorObs(_wallHit); //bool = 1
            AddVectorObs(_wallHitTimer); //float = 1

            /* Model_07 */
            AddVectorObs(_isIdle); //bool = 1
            AddVectorObs(_idleTimer); //float = 1
            AddVectorObs(_lastPosition); //Vector3 = 3
            AddVectorObs((nextCollider.transform.position - transform.position).sqrMagnitude); //float = 1
            if((nextCollider.transform.position - transform.position).sqrMagnitude < (nextCollider.transform.position - _lastPosition).sqrMagnitude) //When distance is closer than last distance to checkpoint
            {
                AddReward(_movedTowardsPointReward);
            }
        }

        public override void AgentAction(float[] vectorAction)
        {
            InterpretDiscreteActions(vectorAction);

            // Find the next checkpoint when registering the current checkpoint that the agent has passed.
            int next = (checkpointIndex + 1) % Colliders.Length;
            Collider nextCollider = Colliders[next];
            Vector3 direction = (nextCollider.transform.position - kart.transform.position).normalized;
            float reward = Vector3.Dot(kart.Rigidbody.velocity.normalized, direction);

            if (ShowRaycasts)
            {
                Debug.DrawRay(AgentSensorTransform.position, kart.Rigidbody.velocity, Color.blue);
            }

            // Add rewards if the agent is heading in the right direction
            AddReward(reward * TowardsCheckpointReward);

            /* Model_07 punish when standing idle */
            if (_newPositionTimer > _newPositionAfter)
            {
                _newPositionTimer = 0.0f;
                _lastPosition = transform.position;
            }

            if(transform.position.x < _lastPosition.x + 1.0f && transform.position.x > _lastPosition.x - 1.0f
                && transform.position.z < _lastPosition.z + 1.0f && transform.position.z > _lastPosition.z - 1.0f)
            {
                _isIdle = true;
            }
            else
            {
                _isIdle = false;
                _idleTimer = 0.0f;
            }

            if(_idleTimer > _resetIdleTimerAfter)
            {
                AddReward(_idlePunishment);
                Done();
                AgentReset();
            }
            //AddReward(kart.LocalSpeed() * SpeedReward);
        }

        public override void AgentReset()
        {
            switch (Mode)
            {
                case AgentMode.Training:
                    /* Random checkpoint selection */
                    checkpointIndex = checkpointIndex = Random.Range(0, Colliders.Length - 1);
                    Collider collider = Colliders[checkpointIndex];
                    transform.localRotation = collider.transform.rotation;
                    transform.position = collider.transform.position;
                    /* Starting at model_05: new way of training 
                    transform.position = _startPosition.position; 
                    transform.localRotation = Quaternion.identity;
                    checkpointIndex = (Colliders.Length - 1); // Set the start index to the index behind the starting position, which is the last checkpoint out of all checkpoints!! 
                     */
                    kart.Rigidbody.velocity = default;
                    acceleration = 0f;
                    steering = 0f;
                    /* Rewarding model_05
                    _roundStarted = false;
                    _timer = 0.0f;
                     */

                    /* Rewarding model_06*/
                    _wallHit = false;
                    _wallHitTimer = 0.0f;

                    /* Rewarding model_07*/
                    _isIdle = false;
                    _idleTimer = 0.0f;
                    _newPositionTimer = 0.0f;
                    _lastPosition = transform.position;
                    break;
                default:
                    break;
            }
        }

        public override float[] Heuristic()
        {
            localActions[0] = Input.GetAxis("Horizontal") + 1;
            localActions[1] = Sign(Input.GetAxis("Vertical"));
            return localActions;
        }

        public Vector2 GenerateInput()
        {
            return new Vector2(steering, acceleration);
        }
    }
}
