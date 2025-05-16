#!/usr/bin/env python3
import asyncio
import logging
import websockets
import msgpack
import msgpack_numpy
import rospy
import numpy as np

# Initialize ROS node
# rospy.init_node('dummy_server', anonymous=True) #removed cause this should only be on ros nodes

async def handle_client(websocket, path):
    logging.info("Client connected")

    # Send metadata (empty for now)
    await websocket.send(msgpack.packb({}))

    try:
        while True:
            # Receive data from the client
            data = await websocket.recv()
            data_dict = msgpack_numpy.unpackb(data)

            # access data
            joint_states = data_dict.get("observation/state")
            gazebo_image = data_dict.get("observation/image")
            wrist_image = data_dict.get("observation/wrist_image")
            depth_image = data_dict.get("observation/depth_image")
            gripper_state = data_dict.get("observation/gripper_state")

            # generate actions (action_chunk)
            # Example values, 1 is for closed and 0 for open
            # we are adding the gripper position to the end of the array
            action_chunk = np.array([[0.5, -0.7, 0.8, 0.5, 0.3, 0.9, 0.2, 1]]) 
            
            #serialize and send the trajectory
            serialized_actions = msgpack.packb({"actions": action_chunk})
            await websocket.send(serialized_actions)
    except websockets.ConnectionClosedOK:
        logging.info("Client disconnected")


async def main():
    logging.basicConfig(level=logging.INFO)
    start_server = websockets.serve(handle_client, "localhost", 8000)
    await start_server
    await asyncio.Future()  # Keep the server running

if __name__ == "__main__":
    asyncio.run(main())
