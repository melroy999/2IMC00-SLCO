/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.Initial;
import slco.SlcoFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Initial</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class InitialTest extends VertexTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(InitialTest.class);
	}

	/**
	 * Constructs a new Initial test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public InitialTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Initial test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected Initial getFixture() {
		return (Initial)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createInitial());
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#tearDown()
	 * @generated
	 */
	@Override
	protected void tearDown() throws Exception {
		setFixture(null);
	}

} //InitialTest
