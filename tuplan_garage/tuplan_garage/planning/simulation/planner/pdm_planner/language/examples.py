import os
import glob
import dspy
from dspy.teleprompt import LabeledFewShot


EXAMPLES_PATH = '/zfsauton2/home/brianyan/tuplan_garage/tuplan_garage/planning/simulation/planner/pdm_planner/language/demonstrations/'


def load_examples(path=EXAMPLES_PATH):
    # Load instructions
    instructions_path = os.path.join(path, 'instructions.txt')
    with open(instructions_path, 'r') as f:
        instructions = f.read()
    instructions = instructions.split('\n')

    # Load demos
    demo_paths = glob.glob(f'{path}/demo*.py')
    demo_paths = sorted(demo_paths, key=lambda x: int(x.split('/')[-1][4:-3]))
    demos = []
    for demo_path in demo_paths:
        with open(demo_path, 'r') as f:
            demo = f.read()
        demos.append(demo)

    assert len(instructions) == len(demos), \
        f'Different number of demos and instructions, got {len(demos)} and {len(instructions)} respectively'
    
    # Build examples
    examples = [dspy.Example(instruction=instruction, code=demo) for (instruction, demo) in zip(instructions, demos)]
    return examples


class Instruction(dspy.Signature):
    instruction = dspy.InputField()
    code=dspy.OutputField()


class InstructionToCode(dspy.Module):
    def __init__(self):
        super().__init__()
        
        self.predict = dspy.Predict(Instruction)

    def forward(self, instruction):
        prediction = self.predict(instruction=instruction)
        return dspy.Prediction(code=prediction.code)


if __name__ == '__main__':
    examples = load_examples()

    turbo = dspy.OpenAI(model='gpt-3.5-turbo', temperature=0.0)
    dspy.settings.configure(lm=turbo)

    predict = InstructionToCode()
    tp = LabeledFewShot()
    predict_comp = tp.compile(predict, trainset=examples)
    pred = predict_comp(instruction='Stay in the current lane.')
    print(pred.code)
