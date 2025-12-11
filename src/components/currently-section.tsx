import Image from "next/image";

const learningTopics = [
  
  
  {
    topic: "Full stack(Java)",
    imagePath: "/full-stack-developer.gif",
  },
  {
    topic: "AI ML",
    imagePath: "/ai ml1.gif",
  },
  {
    topic: "AWS Cloud",
    imagePath: "/Cloud-gif-2.gif",
  },
];

export function CurrentlySection() {
  return (
    <section className="py-16">
      <h2 className="text-3xl font-bold text-center mb-12">
        What I&apos;m currently learning
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
        {learningTopics.map((topic, index) => (
          <div key={index} className="flex flex-col items-center">
            <div className="relative w-64 h-64 rounded-xl overflow-hidden border border-primary/20 shadow-lg">
              <Image
                src={topic.imagePath}
                width={500}
                height={500}
                alt={topic.topic}
                unoptimized={true}
              />
            </div>
            <h3 className="mt-4 text-xl font-semibold">{topic.topic}</h3>
          </div>
        ))}
      </div>
    </section>
  );
}
