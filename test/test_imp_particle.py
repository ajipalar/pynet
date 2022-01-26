import IMP
model = IMP.Model()
particle_0 = model.add_particle("my first particle")
string_key = IMP.StringKey("my first data")
model.add_attribute(string_key, particle_0, "Hi, particle 0")

particle_1 = model.add_particle("my second particle :)")
model.add_attribute(string_key, particle_1, "Hi, particle 1")

print(model.get_attribute(string_key, particle_0))
print(model.get_attribute(string_key, particle_1))

pi_list = model.get_particle_indexes()
p0 = pi_list[0]
p1 = pi_list[1]

print(f"particle 0: {model.get_particle(p0)}")
print(f"particle 1: {model.get_particle(p1)}")
