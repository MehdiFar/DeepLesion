class LesionFinding(object):
  '''
  Represents a lesion
  '''
  
  def __init__(self, lesionid=None, x_min=None, y_min=None, x_max=None, y_max=None, coordType="px",
               CADprobability=None, lesionType=None, state=None, seriesInstanceUID=None):

    # set the variables and convert them to the correct type
    self.id = lesionid
    self.x_min = x_min
    self.y_min = y_min
    self.x_max = x_max
    self.y_max = y_max
    self.CADprobability = CADprobability
    self.lesionType = lesionType
    self.state = state
    self.candidateID = None
    self.seriesuid = seriesInstanceUID
