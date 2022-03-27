import pygame as pyg
import numpy as np
#initialize pygame
pyg.init()
pyg.display.set_caption('Modified PONG')

#resolution and fps
WIDTH, HEIGHT = 1280, 720
window = pyg.display.set_mode((WIDTH, HEIGHT))
FPS = 240


#some basic rgb color codes
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)


SCORE_FONT = pyg.font.SysFont("Verdana", 50)

pad_width = 30
pad_height = 150
ballsize = 15

class Paddle:
	COLOR = RED
	velo = 5
	angvelo = 1 #1 degree per frame generated
	cos = np.cos(np.pi*angvelo/180)
	sin = np.sin(np.pi*angvelo/180)


	def __init__(self, x, y, width, height, angle):
		self.x = self.origx = x
		self.y = self.origy = y
		self.center = np.array([self.x,self.y])
		self.width = width
		self.height = height
		self.angle = self.origangle = angle

		#define corners as displacement from the center
		self.p0disp = self.origp0disp = np.array([-self.width//2, -self.height//2]) #initially top left
		self.p1disp = self.origp1disp = np.array([self.width//2, -self.height//2]) #intially top right
		self.p2disp = self.origp2disp = np.array([self.width//2, +self.height//2]) #initially bottom right
		self.p3disp = self.origp3disp = np.array([-self.width//2, +self.height//2]) #initailly bottom left
		self.quadrant = self.origquadrant = 1
		self.pdisps = [self.p0disp, self.p1disp, self.p2disp, self.p3disp]
		



	'''
		def keepInBounds(self):
			for point in self.points:
				disp = 0
				if point[1] < 0:
					disp =  -1*point[1]

				elif point[1] > HEIGHT:
					disp = HEIGHT - point[1]
	'''


	def draw(self, win):
		#accounts for any rotation and displacement
		p0 = (int(self.x + self.p0disp[0]), int(self.y + self.p0disp[1]))
		p1 = (int(self.x + self.p1disp[0]), int(self.y + self.p1disp[1]))
		p2 = (int(self.x + self.p2disp[0]), int(self.y + self.p2disp[1]))
		p3 = (int(self.x + self.p3disp[0]), int(self.y + self.p3disp[1]))
		self.points = [p0,p1,p2,p3]
		#self.keepInBounds()
		pyg.draw.polygon(win, self.COLOR, self.points, width = 0)

	def move(self, up = True):
		if up:
			self.y -= self.velo
		else:
			self.y += self.velo
			
		self.center = np.array([self.x, self.y])



	def rot(self, cw = True):

		if cw:
			rotMatrix = np.array([[self.cos, -1*self.sin],[self.sin, self.cos]])
			self.p0disp = np.dot(rotMatrix, self.p0disp)
			self.p1disp = np.dot(rotMatrix, self.p1disp)
			self.p2disp = np.dot(rotMatrix, self.p2disp)
			self.p3disp = np.dot(rotMatrix, self.p3disp)
			self.pdisps = [self.p0disp, self.p1disp, self.p2disp, self.p3disp]

			#check if we need to update quadrant for collision system
			if self.angle > 0:
				if (self.angle+ self.angvelo) % 90 < self.angle % 90:
					#we have changed quadrants 
					self.quadrant = (self.quadrant % 4) + 1

			if self.angle < 0: 
				if (self.angle % -90) + self.angvelo > 0:
					#we have changed quadrants
					self.quadrant = (self.quadrant % 4) + 1


			self.angle += self.angvelo
			


		else:
			rotMatrix = np.array([[self.cos, self.sin],[-1*self.sin, self.cos]])
			self.p0disp = np.dot(rotMatrix, self.p0disp).tolist()
			self.p1disp = np.dot(rotMatrix, self.p1disp).tolist()
			self.p2disp = np.dot(rotMatrix, self.p2disp).tolist()
			self.p3disp = np.dot(rotMatrix, self.p3disp).tolist()
			self.pdisps = [self.p0disp, self.p1disp, self.p2disp, self.p3disp]

			if self.angle < 0:
				if (self.angle - self.angvelo) % -90 > self.angle % -90: 
					#we have swapped quadrants
					self.quadrant = self.quadrant - 1
					if self.quadrant == 0:
						self.quadrant = 4
			if self.angle >= 0:
				if self.angle % 90 -self.angvelo < 0: #open because lower bound of quadrant is still same quadrant
					self.quadrant = self.quadrant - 1
					if self.quadrant == 0:
						self.quadrant = 4

			self.angle -= self.angvelo
	def reset(self):
		self.x = self.origx
		self.y = self.origy
		self.p0disp = self.origp0disp
		self.p1disp = self.origp1disp
		self.p2disp = self.origp2disp
		self.p3disp = self.origp3disp
		self.angle = self.origangle
		self.quadrant = self.origquadrant
		self.pdisps = [self.p0disp, self.p1disp, self.p2disp, self.p3disp]



class Ball:
	VELO = -2
	COLOR = BLUE

	def __init__(self, x, y, size):
		self.x = self.origx = x
		self.y = self.origy = y
		self.size = size
		self.xvelo = self.VELO
		self.yvelo = 0

	def draw(self, win):
		pyg.draw.circle(win, self.COLOR, (self.x, self.y), self.size)

	def move(self):
		self.x += self.xvelo
		self.y += self.yvelo

	def reset(self):
		self.x = self.origx
		self.y = self.origy
		self.xvelo = -1*self.xvelo*np.abs(self.VELO)/np.abs(self.xvelo) #prefactor -1 to make the player who scores serve
		self.yvelo = 0



#want to draw each object in the frame
def draw(win, paddles, ball, left_score, right_score, winner, winning_text, new_game):
	win.fill(WHITE)

	left_score_txt  = SCORE_FONT.render(f"{left_score}", 1, BLACK)
	right_score_txt  = SCORE_FONT.render(f"{right_score}", 1, BLACK)
	win.blit(left_score_txt, (WIDTH//4 - left_score_txt.get_width()//2, 30))
	win.blit(right_score_txt, (3*WIDTH//4 - right_score_txt.get_width()//2, 30))
	if winner:
		winning_txt  = SCORE_FONT.render(f"{winning_text}", 1, BLACK)
		window.blit(winning_txt, (WIDTH//2 - winning_txt.get_width()//2, 60))
	if new_game:
		new_game_txt = SCORE_FONT.render(f"{'Resetting the Game'}", 1, BLACK)
		window.blit(new_game_txt, (WIDTH//2 - new_game_txt.get_width()//2, 60))

	for paddle in paddles:
		paddle.draw(win)
	ball.draw(win)
	pyg.display.update()



#update the paddle position based on key press
def handle_paddle_movement(keys, left, right):

	#moves the two paddles up and down
	quadmaxy = [2,1,0,3]
	quadminy = [0,3,2,1]

	if keys[pyg.K_w] and left.y + left.pdisps[quadminy[left.quadrant-1]][1] - left.velo >= 0:
		left.move(up = True)
	if keys[pyg.K_s] and left.y + left.pdisps[quadmaxy[left.quadrant-1]][1] + left.velo <= HEIGHT:
		left.move(up = False)
	if keys[pyg.K_UP] and right.y + right.pdisps[quadminy[right.quadrant-1]][1] - right.velo >= 0:
		right.move(up = True)
	if keys[pyg.K_DOWN] and right.y + right.pdisps[quadmaxy[right.quadrant-1]][1] + right.velo <= HEIGHT:
		right.move(up = False)


	#rotate the paddles cw and ccw
	if keys[pyg.K_d]:
		left.rot(cw = True)
	if keys[pyg.K_a]:
		left.rot(cw = False)
	if keys[pyg.K_RIGHT]:
		right.rot(cw = True)
	if keys[pyg.K_LEFT]:
		right.rot(cw = False)





#determine the angle between the ball and the paddle, modifies the balls velocity vector according to elastic collision

def handle_ball_collision(ball, leftpad, rightpad):
	#check if we collide with the box boundaries
	if ball.y + ball.size >= HEIGHT:
		ball.yvelo *= -1
	elif ball.y - ball.size <= 0: 
		ball.yvelo *= -1



	#store the bounds of the paddle, when ball is in this wider box, we check for collisions
	quadmaxx = [1,0,3,2] #these values assign a label to each point given the rotation angle (point 1 is the maximum x bound in quadrant 1 for example) 
	quadmaxy = [2,1,0,3]
	quadminx = [3,2,1,0]
	quadminy = [0,3,2,1]

	#is the ball on the left or right side of the screen - tells us which paddle to check
	collision_check = False

	if ball.x >= WIDTH//2:
		quadrant = rightpad.quadrant - 1
		#is any part of the ball within the x-y box defining the bounds of the paddle?

		if ball.x + ball.size >= rightpad.points[quadminx[quadrant]][0] and ball.x - ball.size <= rightpad.points[quadmaxx[quadrant]][0]:
			#ball is in the x range of the box
			if ball.y + ball.size >= rightpad.points[quadminy[quadrant]][1] and ball.y - ball.size <= rightpad.points[quadmaxy[quadrant]][1]:
				#ball is within the entire box we wish to check
				collision_check = True

		if collision_check:
			#can check which corner of the paddle box the ball is in using a quadrant dependent comparison
			ax = np.array(rightpad.points[quadmaxx[quadrant]])
			ay = np.array(rightpad.points[quadmaxy[quadrant]])
			ix = np.array(rightpad.points[quadminx[quadrant]])
			iy = np.array(rightpad.points[quadminy[quadrant]])	


			if rightpad.angle%90 == 0: #we will have vertical lines, easy to check
				#ball is to the left of the paddle, contacting on the right
				if ball.x <= ix[0] and ball.x + ball.size >= ix[0] and ball.y+0.95*ball.size > iy[1] and ball.y-0.95*ball.size < ay[1] and ball.xvelo >= 0:
					ball.xvelo *= -1
				#ball is to the right of the paddle, contacting on the left
				if ball.x >= ax[0] and ball.x - ball.size <= ax[0] and ball.y+0.95*ball.size > iy[1] and ball.y-0.95*ball.size < ay[1] and ball.xvelo <= 0:
					ball.xvelo*= -1
				#ball is above the paddle contacting from above
				if ball.y <= iy[1] and ball.y + ball.size >= iy[1] and ball.x+0.95*ball.size > ix[0] and ball.x-0.95*ball.size < ax[0]:
					ball.yvelo*= -1
					if ball.yvelo == 0:
						ball.yvelo -= (rightpad.velo+0.01)
				#ball is below the paddle contacting from below
				if ball.y >= ay[1] and ball.y - ball.size <= ay[1] and ball.x+0.95*ball.size > ix[0] and ball.x-0.95*ball.size < ax[0]:
					ball.yvelo*= -1
					if ball.yvelo == 0:
						ball.yvelo += (rightpad.velo+0.01)

			else: #lines are slanted much more difficult
				correct = False #use this to verify side that the ball is hitting

				#######CONVENTION#######
				#ball will always act like it hit the longer side if it hits the corner (can be changed later <= becomes < in the var correct check)

				#REGION 1
				if ball.y > ax[1] + (ay[1]-ax[1])/(ay[0]-ax[0])*(ball.x - ax[0]):
					#ball is in visual lower right region 1
					#double check ball will hit region 1 and not 2 or 4
					if ball.y >= ax[1] and ball.x >= ay[0]: #center of the ball determines whether the designation is correct
						correct = True
					if correct:
						#we can do a real collision detection and know the surface
						ballposvec = np.array([ball.x, ball.y])
						surfacevec = np.array([ay[0]-ax[0], ay[1]-ax[1]])
						normalvec = np.array([surfacevec[1], -1*surfacevec[0]])
						normalvec = normalvec/np.linalg.norm(normalvec)
						ballcontactpos = ballposvec + ball.size*normalvec

						#now check if ball has truly crossed plane of contact, if so we apply the contact modification to the velocity
						if ballcontactpos[1] <= ax[1] + (ay[1]-ax[1])/(ay[0]-ax[0])*(ball.x - ax[0]) or ballcontactpos[0] <= ax[0] + (ay[0]-ax[0])/(ay[1]-ax[1])*(ball.y - ax[1]):
							ballvelovec = np.array([ball.xvelo, ball.yvelo])
							theta = np.arccos(np.dot(surfacevec, ballvelovec)/(np.linalg.norm(surfacevec)*np.linalg.norm(ballvelovec)))
							#clockwise around xy, counter clockwise visually by twice the angle between the vectors
							rotMatrix = np.array([[np.cos(2*theta), np.sin(2*theta)], [-1*np.sin(2*theta), np.cos(2*theta)]])
							ballvelovec = np.dot(rotMatrix, ballvelovec)
							ball.xvelo = ballvelovec[0]
							ball.yvelo = ballvelovec[1]

				#REGION 2
				if ball.y > ay[1] + (ay[1]-ix[1])/(ay[0]-ix[0])*(ball.x - ay[0]):
					#ball is in visual lower left region 2
					#double check ball will hit region 2 and not 1 or 3
					if ball.y > ix[1] and ball.x < ay[0]:
						correct = True
					
					if correct:

						ballposvec = np.array([ball.x, ball.y])
						surfacevec = np.array([ix[0]-ay[0], ix[1]-ay[1]])
						normalvec = np.array([surfacevec[1], -1*surfacevec[0]])
						normalvec = normalvec/np.linalg.norm(normalvec)
						ballcontactpos = ballposvec + ball.size*normalvec


						if ballcontactpos[0] <= ay[1] + (ay[1]-ix[1])/(ay[0]-ix[0])*(ball.x - ay[0]) or ballcontactpos[0] >= ay[0] + (ay[0]-ix[0])/(ay[1]-ix[1])*(ball.y - ay[1]):
							ballvelovec = np.array([ball.xvelo, ball.yvelo])
							theta = np.arccos(np.dot(surfacevec, ballvelovec)/(np.linalg.norm(surfacevec)*np.linalg.norm(ballvelovec)))
							#clockwise around xy, counter clockwise visually by twice the angle between the vectors
							rotMatrix = np.array([[np.cos(2*theta), np.sin(2*theta)], [-1*np.sin(2*theta), np.cos(2*theta)]])
							ballvelovec = np.dot(rotMatrix, ballvelovec)
							ball.xvelo = ballvelovec[0]
							ball.yvelo = ballvelovec[1]

				#REGION 3
				if ball.y < iy[1] + (iy[1]-ix[1])/(iy[0]-ix[0])*(ball.x - iy[0]):
					#ball is in visual upper left region 3
					#double check ball will hit region 3 and not 2 or 4
					if ball.y <= ix[1] and ball.x <= iy[0]:
						correct = True
					if correct:
						ballposvec = np.array([ball.x, ball.y])
						surfacevec = np.array([iy[0]-ix[0], iy[1]-ix[1]])
						normalvec = np.array([surfacevec[1], -1*surfacevec[0]])
						normalvec = normalvec/np.linalg.norm(normalvec)
						ballcontactpos = ballposvec + ball.size*normalvec

						if ballcontactpos[1] >= iy[1] + (iy[1]-ix[1])/(iy[0]-ix[0])*(ball.x - iy[0]) or ballcontactpos[0] >= ix[0] + (iy[0]-ix[0])/(iy[1]-ix[1])*(ball.y - ix[1]):
							ballvelovec = np.array([ball.xvelo, ball.yvelo])
							theta = np.arccos(np.dot(surfacevec, ballvelovec)/(np.linalg.norm(surfacevec)*np.linalg.norm(ballvelovec)))
							#clockwise around xy, counter clockwise visually by twice the angle between the vectors
							rotMatrix = np.array([[np.cos(2*theta), np.sin(2*theta)], [-1*np.sin(2*theta), np.cos(2*theta)]])
							ballvelovec = np.dot(rotMatrix, ballvelovec)
							ball.xvelo = ballvelovec[0]
							ball.yvelo = ballvelovec[1]

				#REGION 4
				if ball.y < ax[1] + (ax[1]-iy[1])/(ax[0]-iy[0])*(ball.x - ax[0]):
					#ball is in visual upper right region 4
					#double check ball will hit region 4 and not 3 or 1
					if ball.y < ax[1] and ball.x > iy[0]:
						correct = True
					if correct:
						ballposvec = np.array([ball.x, ball.y])
						surfacevec = np.array([ax[0]-iy[0], ax[1]-iy[1]])
						normalvec = np.array([surfacevec[1], -1*surfacevec[0]])
						normalvec = normalvec/np.linalg.norm(normalvec)
						ballcontactpos = ballposvec + ball.size*normalvec

						if ballcontactpos[1] >= ax[1] + (ax[1]-iy[1])/(ax[0]-iy[0])*(ball.x - ax[0]) or ballcontactpos[0] <= ax[0] + (ax[0]-iy[0])/(ax[1]-iy[1])*(ball.y - ax[1]):
							ballvelovec = np.array([ball.xvelo, ball.yvelo])
							theta = np.arccos(np.dot(surfacevec, ballvelovec)/(np.linalg.norm(surfacevec)*np.linalg.norm(ballvelovec)))
							#clockwise around xy, counter clockwise visually by twice the angle between the vectors
							rotMatrix = np.array([[np.cos(2*theta), np.sin(2*theta)], [-1*np.sin(2*theta), np.cos(2*theta)]])
							ballvelovec = np.dot(rotMatrix, ballvelovec)
							ball.xvelo = ballvelovec[0]
							ball.yvelo = ballvelovec[1]

	else: #ball is by the left paddle now
		quadrant = leftpad.quadrant - 1
		#is the ball close enough to hit the paddle?
		if ball.x + ball.size >= leftpad.points[quadminx[quadrant]][0] and ball.x - ball.size <= leftpad.points[quadmaxx[quadrant]][0]:
			#ball is in the x range of the box
			if ball.y + ball.size >= leftpad.points[quadminy[quadrant]][1] and ball.y - ball.size <= leftpad.points[quadmaxy[quadrant]][1]:
				#ball is within the entire box we wish to check
				collision_check = True

		if collision_check:
			#can check which corner of the paddle box the ball is in using a quadrant dependent comparison
			ax = np.array(leftpad.points[quadmaxx[quadrant]])
			ay = np.array(leftpad.points[quadmaxy[quadrant]])
			ix = np.array(leftpad.points[quadminx[quadrant]])
			iy = np.array(leftpad.points[quadminy[quadrant]])	

			if leftpad.angle%90 == 0: #we will have vertical lines, easy to check
				if ball.x <= ix[0] and ball.x + ball.size >= ix[0] and ball.y+0.95*ball.size > iy[1] and ball.y-0.95*ball.size < ay[1] and ball.xvelo >= 0:
					ball.xvelo *= -1
				if ball.x >= ax[0] and ball.x - ball.size <= ax[0] and ball.y+0.95*ball.size > iy[1] and ball.y-0.95*ball.size < ay[1] and ball.xvelo <= 0:
					ball.xvelo*= -1
				if ball.y <= iy[1] and ball.y + ball.size >= iy[1] and ball.x+0.95*ball.size > ix[0] and ball.x-0.95*ball.size < ax[0]:
					ball.yvelo*= -1
					if ball.yvelo == 0:
						ball.yvelo -= (leftpad.velo+0.1)
				if ball.y >= ay[1] and ball.y - ball.size <= ay[1] and ball.x+0.95*ball.size > ix[0] and ball.x-0.95*ball.size < ax[0]:
					ball.yvelo*= -1
					if ball.yvelo == 0:
						ball.yvelo += (leftpad.velo+0.1)

			else: #lines are slanted much more difficult
				correct = False #use this to verify side that the ball is hitting

				#######CONVENTION#######
				#ball will always act like it hit the longer side if it hits the corner (can be changed later <= becomes < in the var correct check)

				#REGION 1
				if ball.y > ax[1] + (ay[1]-ax[1])/(ay[0]-ax[0])*(ball.x - ax[0]):
					#ball is in visual lower right region 1
					#double check ball will hit region 1 and not 2 or 4
					if ball.y >= ax[1] and ball.x >= ay[0]: #center of the ball determines whether the designation is correct
						correct = True
					if correct:
						#we can do a real collision detection and know the surface
						ballposvec = np.array([ball.x, ball.y])
						surfacevec = np.array([ay[0]-ax[0], ay[1]-ax[1]])
						normalvec = np.array([surfacevec[1], -1*surfacevec[0]])
						normalvec = normalvec/np.linalg.norm(normalvec)
						ballcontactpos = ballposvec + ball.size*normalvec

						#now check if ball has truly crossed plane of contact, if so we apply the contact modification to the velocity
						if ballcontactpos[1] <= ax[1] + (ay[1]-ax[1])/(ay[0]-ax[0])*(ball.x - ax[0]) or ballcontactpos[0] <= ax[0] + (ay[0]-ax[0])/(ay[1]-ax[1])*(ball.y - ax[1]):
							ballvelovec = np.array([ball.xvelo, ball.yvelo])
							theta = np.arccos(np.dot(surfacevec, ballvelovec)/(np.linalg.norm(surfacevec)*np.linalg.norm(ballvelovec)))
							#clockwise around xy, counter clockwise visually by twice the angle between the vectors
							rotMatrix = np.array([[np.cos(2*theta), np.sin(2*theta)], [-1*np.sin(2*theta), np.cos(2*theta)]])
							ballvelovec = np.dot(rotMatrix, ballvelovec)
							ball.xvelo = ballvelovec[0]
							ball.yvelo = ballvelovec[1]

				#REGION 2
				if ball.y > ay[1] + (ay[1]-ix[1])/(ay[0]-ix[0])*(ball.x - ay[0]):
					#ball is in visual lower left region 2
					#double check ball will hit region 2 and not 1 or 3
					if ball.y > ix[1] and ball.x < ay[0]:
						correct = True
					
					if correct:

						ballposvec = np.array([ball.x, ball.y])
						surfacevec = np.array([ix[0]-ay[0], ix[1]-ay[1]])
						normalvec = np.array([surfacevec[1], -1*surfacevec[0]])
						normalvec = normalvec/np.linalg.norm(normalvec)
						ballcontactpos = ballposvec + ball.size*normalvec


						if ballcontactpos[0] <= ay[1] + (ay[1]-ix[1])/(ay[0]-ix[0])*(ball.x - ay[0]) or ballcontactpos[0] >= ay[0] + (ay[0]-ix[0])/(ay[1]-ix[1])*(ball.y - ay[1]):
							ballvelovec = np.array([ball.xvelo, ball.yvelo])
							theta = np.arccos(np.dot(surfacevec, ballvelovec)/(np.linalg.norm(surfacevec)*np.linalg.norm(ballvelovec)))
							#clockwise around xy, counter clockwise visually by twice the angle between the vectors
							rotMatrix = np.array([[np.cos(2*theta), np.sin(2*theta)], [-1*np.sin(2*theta), np.cos(2*theta)]])
							ballvelovec = np.dot(rotMatrix, ballvelovec)
							ball.xvelo = ballvelovec[0]
							ball.yvelo = ballvelovec[1]

				#REGION 3
				if ball.y < iy[1] + (iy[1]-ix[1])/(iy[0]-ix[0])*(ball.x - iy[0]):
					#ball is in visual upper left region 3
					#double check ball will hit region 3 and not 2 or 4
					if ball.y <= ix[1] and ball.x <= iy[0]:
						correct = True
					if correct:
						ballposvec = np.array([ball.x, ball.y])
						surfacevec = np.array([iy[0]-ix[0], iy[1]-ix[1]])
						normalvec = np.array([surfacevec[1], -1*surfacevec[0]])
						normalvec = normalvec/np.linalg.norm(normalvec)
						ballcontactpos = ballposvec + ball.size*normalvec

						if ballcontactpos[1] >= iy[1] + (iy[1]-ix[1])/(iy[0]-ix[0])*(ball.x - iy[0]) or ballcontactpos[0] >= ix[0] + (iy[0]-ix[0])/(iy[1]-ix[1])*(ball.y - ix[1]):
							ballvelovec = np.array([ball.xvelo, ball.yvelo])
							theta = np.arccos(np.dot(surfacevec, ballvelovec)/(np.linalg.norm(surfacevec)*np.linalg.norm(ballvelovec)))
							#clockwise around xy, counter clockwise visually by twice the angle between the vectors
							rotMatrix = np.array([[np.cos(2*theta), np.sin(2*theta)], [-1*np.sin(2*theta), np.cos(2*theta)]])
							ballvelovec = np.dot(rotMatrix, ballvelovec)
							ball.xvelo = ballvelovec[0]
							ball.yvelo = ballvelovec[1]

				#REGION 4
				if ball.y < ax[1] + (ax[1]-iy[1])/(ax[0]-iy[0])*(ball.x - ax[0]):
					#ball is in visual upper right region 4
					#double check ball will hit region 4 and not 3 or 1
					if ball.y < ax[1] and ball.x > iy[0]:
						correct = True
					if correct:
						ballposvec = np.array([ball.x, ball.y])
						surfacevec = np.array([ax[0]-iy[0], ax[1]-iy[1]])
						normalvec = np.array([surfacevec[1], -1*surfacevec[0]])
						normalvec = normalvec/np.linalg.norm(normalvec)
						ballcontactpos = ballposvec + ball.size*normalvec

						if ballcontactpos[1] >= ax[1] + (ax[1]-iy[1])/(ax[0]-iy[0])*(ball.x - ax[0]) or ballcontactpos[0] <= ax[0] + (ax[0]-iy[0])/(ax[1]-iy[1])*(ball.y - ax[1]):
							ballvelovec = np.array([ball.xvelo, ball.yvelo])
							theta = np.arccos(np.dot(surfacevec, ballvelovec)/(np.linalg.norm(surfacevec)*np.linalg.norm(ballvelovec)))
							#clockwise around xy, counter clockwise visually by twice the angle between the vectors
							rotMatrix = np.array([[np.cos(2*theta), np.sin(2*theta)], [-1*np.sin(2*theta), np.cos(2*theta)]])
							ballvelovec = np.dot(rotMatrix, ballvelovec)
							ball.xvelo = ballvelovec[0]
							ball.yvelo = ballvelovec[1]

	#elimate boring game conditions manage this later
	#if np.abs(ball.xvelo) < 0.5:
		#if ball.xvelo < 0:
			#ball.xvelo = -0.5
		#if ball.xvelo > 0:
			#ball.xvelo = 0.5
				



#main while loop to run the game
def main():
	run = True
	clock = pyg.time.Clock() #will define the game's max fps


	leftPad = Paddle(90, HEIGHT//2, pad_width, pad_height, 0)
	rightPad = Paddle(WIDTH - 90 , HEIGHT//2, pad_width, pad_height, 0)
	ball = Ball(WIDTH//2, HEIGHT//2, ballsize)

	left_score = 0
	right_score = 0
	delay = 0

	winning_score = 3

	winner = False
	winning_text = ''
	new_game = False
	while run:
		clock.tick(FPS) #caps the run rate of the while loop to FPS times/second
		draw(window, [leftPad, rightPad], ball, left_score, right_score, winner, winning_text, new_game) # redraw the window

		for event in pyg.event.get():
			if event.type == pyg.QUIT:
				run = False
				break

		if delay == 1:
			pyg.time.delay(1000)
			delay = 0

		keys = pyg.key.get_pressed()
		handle_paddle_movement(keys, leftPad, rightPad)
		ball.move()
		handle_ball_collision(ball, leftPad, rightPad)

		if ball.x < 0 :
			right_score += 1
			ball.reset()
			leftPad.reset()
			rightPad.reset()
			delay = 1
			if right_score >= winning_score:
				winning_text = 'Right Player Wins'
				winner = True
		elif ball.x > WIDTH :
			left_score += 1
			ball.reset()
			leftPad.reset()
			rightPad.reset()
			delay = 1
			if left_score >= winning_score:
				winning_text = 'Left Player Wins'
				winner = True
		if winner:
			draw(window, [leftPad, rightPad], ball, left_score, right_score, winner, winning_text, new_game)
			pyg.time.delay(5000)
			left_score = 0
			right_score = 0
			winner = False
			ball.reset()
			leftPad.reset()
			rightPad.reset()
			new_game = True
			draw(window, [leftPad, rightPad], ball, left_score, right_score, winner, winning_text, new_game)
			pyg.time.delay(2000)
			new_game = False
			delay = 1


	pyg.quit()


#only run the game when this particular game file is called in case we want to import game files elsewhere
if __name__ == '__main__':
	main()

