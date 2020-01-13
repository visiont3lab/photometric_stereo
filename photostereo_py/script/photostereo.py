import cv2 as cv
import numpy as np
import vtk

class photometry:

    def __init__(self, numimg, display):
        self.IMAGES = numimg
        self.display = display
        self.normalmap = []
        self.albedo = []
        self.pgrads = []
        self.qgrads = []
        self.gaussgrad = []
        self.meangrad = []
        self.mask = []
        self.Z = []

    def runphotometry(self, input_array, mask=None):
        print("Running main process. Be patient...")
        if (mask is not None):
            self.mask = mask
            for id in range(0, self.IMAGES):
                input_array[id] = np.multiply(input_array[id], mask/255)

        # Convert input array to float img array
        input_arr_conv = []
        for id in range (0, self.IMAGES):
            im_fl = np.float32(input_array[id])
            im_fl = im_fl / 255
            input_arr_conv.append(im_fl)
        h = input_arr_conv[0].shape[0]
        w = input_arr_conv[0].shape[1]
        self.normalmap = np.zeros((h, w, 3), dtype=np.float32)
        self.pgrads = np.zeros((h, w), dtype=np.float32)
        self.qgrads = np.zeros((h, w), dtype=np.float32)
        lpinv = np.linalg.pinv(self.light_mat)
        intensities = []
        norm = []
        for imid in range(0, self.IMAGES):
            a = np.array(input_arr_conv[imid]).reshape(-1)
            intensities.append(a)
        intensities = np.array(intensities)
        rho_z = np.einsum('ij,jk->ik', lpinv, intensities)
        rho = rho_z.transpose()
        norm.append(np.sum(np.abs(rho)**2, axis=-1)**(1./2))
        norm_t = np.array(norm).transpose()
        norm_t = np.clip(norm_t, 0 , 1)
        norm_t = np.where(norm_t==0, 1, norm_t)
        self.albedo = np.reshape(norm_t, (h, w))
        rho = np.divide(rho , norm_t)
        rho[:, 2] = np.where(rho[:, 2] == 0, 1, rho[:, 2])
        rho = np.asarray(rho).transpose()
        self.normalmap[:, :, 0] = np.reshape(rho[0], (h, w))
        self.normalmap[:, :, 1] = np.reshape(rho[1], (h, w))
        self.normalmap[:, :, 2] = np.reshape(rho[2], (h, w))
        self.pgrads[0:h, 0:w] = self.normalmap[:, :, 0] / self.normalmap[:, :, 2]
        self.qgrads[0:h, 0:w] = self.normalmap[:, :, 1] / self.normalmap[:, :, 2]
        self.normalmap = self.normalmap.astype(np.float32)
        self.normalmap = cv.cvtColor(self.normalmap, cv.COLOR_BGR2RGB)
        output_int = cv.normalize(self.normalmap, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
        output_int = cv.bitwise_and(output_int, output_int, mask = mask)
        self.normalmap = cv.bitwise_and(self.normalmap, self.normalmap, mask = mask)
        if self.display:
            cv.imshow('normal_normalized', output_int)
            cv.imshow('albedo', self.albedo)
            cv.imshow('self.pgrads', self.pgrads)
            cv.imshow('self.qgrads', self.qgrads)
            cv.waitKey(0)
            cv.destroyAllWindows()
        print("Normal map computation end ")
        return self.normalmap

    def computegaussian(self):
        print("Computing gaussian curvature. Be patient...")

        kernely = np.array([[-1, -2, -1], [ 0, 0, 0], [1, 2, 1]], dtype=np.float32)
        kernelx = np.array([[1, 0, -1], [ 2, 0, -2], [1, 0, -1]], dtype=np.float32)

        h = self.pgrads.shape[0]
        w = self.pgrads.shape[1]
        self.gaussgrad = np.zeros((h, w, 1), dtype=np.float32)

        Ixx = cv.filter2D(self.pgrads, cv.CV_32F, kernelx)
        Ixy = cv.filter2D(self.pgrads, cv.CV_32F, kernely)
        Iyy = cv.filter2D(self.qgrads, cv.CV_32F, kernely)
        Iyx = cv.filter2D(self.qgrads, cv.CV_32F, kernelx)

        self.gaussgrad = ((Ixx * Iyy) - Ixy * Iyx) / np.power((1 + np.float_power(self.pgrads, 2) + np.float_power(self.qgrads, 2)), 2)

        print("Gaussian curvature computation end.")
        gaussgrad_norm = cv.normalize(self.gaussgrad, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        if self.display:
            cv.imshow('gaussgrad', gaussgrad_norm)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return gaussgrad_norm

    def computemedian(self):
        print("Computing median curvature. Be patient...")

        h = self.pgrads.shape[0]
        w = self.pgrads.shape[1]
        self.meangrad = np.zeros((h, w, 1), dtype=np.float32)
        scale = 1
        delta = 0
        ddepth = cv.CV_32F

        Ixx = cv.Sobel(self.pgrads, ddepth, 1, 0, ksize=1, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        Ixy = cv.Sobel(self.pgrads, ddepth, 0, 1, ksize=1, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        Iyy = cv.Sobel(self.qgrads, ddepth, 0, 1, ksize=1, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        Iyx = cv.Sobel(self.qgrads, ddepth, 1, 0, ksize=1, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

        a = (1 + np.float_power(self.pgrads, 2)) * Iyy
        b = self.pgrads * self.qgrads * (Ixy + Iyx)
        c = (1 + np.float_power(self.qgrads, 2)) * Ixx
        d = np.float_power(1 + np.float_power(self.pgrads, 2) + np.float_power(self.qgrads, 2), 3 / 2)
        self.meangrad = (a - b + c) / d

        print("Median curvature computation end.")
        meangrad_norm = cv.normalize(self.meangrad, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        if self.display:
            cv.imshow('meangrad', meangrad_norm)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return meangrad_norm

    def setlightmat(self, light_mat):
        self.light_mat = light_mat
        print("Light matrix set")

    def setlmfromts(self, tilt, slant):
        # todo: add check on tilt and slant size
        self.light_mat = np.zeros((self.IMAGES, 3), dtype=np.float32)
        rads = 180 / np.pi
        for id in range (0 , self.IMAGES):
            self.light_mat[id , 0] = np.cos(tilt[id] / rads)
            self.light_mat[id , 1] = np.sin(tilt[id] / rads)
            self.light_mat[id , 2] = np.cos(slant[id] / rads)
            norm = np.linalg.norm(self.light_mat[id])
            self.light_mat[id] = self.light_mat[id]/norm
        print("Light matrix set from Tilt&Slant")
        print(self.light_mat)

    def settsfromlm(self):
        print("TODO")
        rads = 180 / np.pi
        lightarr = np.asarray(self.light_mat).reshape(-1)
        tiltslant = np.zeros((self.IMAGES * 2), dtype=np.float32)
        for id in range (0, self.IMAGES):
            slant = rads * np.arccos(lightarr[3 * id + 1])
            if (lightarr[3 * id] < 0):
                tilt = rads * np.arctan(lightarr[3 * id + 1]/lightarr[3 * id]) + 180
            else:
                tilt = rads * np.arctan(lightarr[3 * id + 1]/lightarr[3 * id])
            tiltslant[2 * id] = tilt - 180
            tiltslant[2 * id + 1] = slant
        tiltslant = tiltslant.reshape((self.IMAGES, 2))
        return tiltslant

    def getnormalmap(self):
        return self.normalmap

    def getalbedo(self):
        return self.albedo

    def computedepthmap(self):
        h = self.normalmap.shape[0]
        w = self.normalmap.shape[1]
        P = np.zeros((h, w, 2), dtype=np.float32)
        Q = np.zeros((h, w, 2), dtype=np.float32)
        tempZ = np.zeros((h, w, 2), dtype=np.float32)
        self.Z = np.zeros((h, w), dtype=np.float32)
        landa = 1.0
        mu = 1.0
        cv.dft(self.pgrads, P, cv.DFT_COMPLEX_OUTPUT)
        cv.dft(self.qgrads, Q, cv.DFT_COMPLEX_OUTPUT)

        for i in range (1, h):
            for j in range (1, w):
                u = np.sin(i * 2 * np.pi / h)
                v = np.sin(j * 2 * np.pi / w)
                uv = np.float_power(u, 2) + np.float_power(v, 2)
                d = (1 + landa)*uv + mu*np.float_power(uv, 2)
                tempZ [i, j, 0] = (u*P[i, j, 1] + v*Q[i, j, 1]) / d
                tempZ [i, j, 1] = (-u*P[i, j, 0] - v*Q[i, j, 0]) / d
        tempZ[0, 0, 0] = 0
        tempZ[0, 0, 1] = 0
        flags = cv.DFT_INVERSE + cv.DFT_SCALE + cv.DFT_REAL_OUTPUT
        cv.dft(tempZ, self.Z, flags)
        z_norm = cv.normalize(self.Z, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        if self.display:
            cv.imshow('z_norm', z_norm)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return z_norm

    def computedepth2(self):
        print("Experimental")
        h = self.normalmap.shape[0]
        w = self.normalmap.shape[1]
        Z = np.zeros((h, w), dtype=np.float32)
        A = np.array([(1, -1, 0),(1, 0, -1)], dtype=np.float32)
        hh = 10
        print(A)
        Apinv = np.linalg.pinv(A)
        print(Apinv)
        for i in range (0, h-1):
            for j in range (0, w-1):
                arr = np.array([-self.pgrads[i,j],-self.qgrads[i,j]], dtype=np.float32)
                temp = np.einsum('ji,i->j', Apinv,arr)
                temp = np.absolute(temp)*hh
                Z[i, j] = temp[0]
                Z[i + 1, j] = temp[1]
                Z[i, j + 1] = temp[2]
        #Z = cv.bitwise_not(Z)
        #self.Z = cv.normalize(Z, None, 0, 10, cv.NORM_MINMAX, cv.CV_32FC1)
        self.Z = np.clip(Z, -10, 10)
        Znorm = cv.normalize(Z, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
        cv.imshow('Znorm', Znorm)
        cv.imshow('Z', Z)
        cv.waitKey(0)
        cv.destroyAllWindows()


    def display3dobj(self):
        colors = vtk.vtkNamedColors()

        h = self.normalmap.shape[0]
        w = self.normalmap.shape[1]

        # Create a triangle
        points = vtk.vtkPoints()
        for x in range (0, h):
            for y in range (0, w):
                points.InsertNextPoint(x, y, self.Z[x, y])

        triangle = vtk.vtkTriangle()
        triangles = vtk.vtkCellArray()
        for i in range (0, h):
            for j in range (0, w):
                triangle.GetPointIds().SetId(0, j + (i * w))
                triangle.GetPointIds().SetId(1, (i + 1) * w + j)
                triangle.GetPointIds().SetId(2, j + (i * w) + 1)
                triangles.InsertNextCell(triangle)
                triangle.GetPointIds().SetId(0, (i + 1)*w + j)
                triangle.GetPointIds().SetId(1, (i + 1)*w + j + 1)
                triangle.GetPointIds().SetId(2, j + (i*w) + 1)
                triangles.InsertNextCell(triangle)


        # Create a polydata object
        trianglePolyData = vtk.vtkPolyData()

        # Add the geometry and topology to the polydata
        trianglePolyData.SetPoints(points)
        trianglePolyData.SetPolys(triangles)

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(trianglePolyData)
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(colors.GetColor3d("Cyan"))
        actor.SetMapper(mapper)

        # Create a renderer, render window, and an interactor
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetWindowName("Triangle")
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Add the actors to the scene
        renderer.AddActor(actor)
        renderer.SetBackground(colors.GetColor3d("DarkGreen"))

        # Render and interact
        renderWindow.Render()
        renderWindowInteractor.Start()

        filename = "test"
        # Write the stl file to disk
        stlWriter = vtk.vtkOBJExporter()
        stlWriter.SetFilePrefix(filename)
        stlWriter.SetInput(renderWindow)
        stlWriter.Write()
